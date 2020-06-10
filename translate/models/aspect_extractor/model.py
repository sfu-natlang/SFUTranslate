"""
The implementation of Aspect Integration Module which combines the concatenation of aspect vectors (linguistic embedding)
 with vanilla transformer token embeddings
"""
import torch
from torch import nn
from torchtext import data

from configuration import cfg, device
from models.transformer.model import Transformer
from readers.data_provider import src_tokenizer_obj

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import BertForMaskedLM
except ImportError:
    warnings.warn("transformers package is not available, transformers.BertForMaskedLM will not be accessible.")
    BertForMaskedLM = None


class AspectIntegration(nn.Module):
    def __init__(self, d_model):
        super(AspectIntegration, self).__init__()
        self.d_model = d_model
        self.bert_lm = None
        self.aspect_vectors = None
        self.linguistic_embedding_to_d_model = None
        self.input_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.softmax = nn.Softmax(dim=-1)
        self.bert_weights_for_average_pooling = None
        # self.number_of_bert_layers = 0

    def init_model_params(self):
        print("Running the init params for BERT language model")
        self.bert_lm = BertForMaskedLM.from_pretrained(src_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        # self.number_of_bert_layers = len(self.bert_lm.bert.encoder.layer) + 1
        # self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers), requires_grad=True)
        print("Running the init params for aspect_vectors")
        try:
            so = torch.load(cfg.aspect_vectors_data_address, map_location=lambda storage, loc: storage)
        except KeyError:
            print("AspectIntegration module failed to initialize since \"aspect_vectors_data_address\" is not set in the config file.\nExiting!")
            exit()
        self.aspect_vectors = so['aspect_vectors'].to(device)
        self.bert_weights_for_average_pooling = nn.Parameter(so['bert_weights'].to(device), requires_grad=True)
        aspect_vector_key_size = self.aspect_vectors[0].out_features
        aspect_vector_feature_count = len(self.aspect_vectors) - 1
        if aspect_vector_feature_count > 0:
            self.linguistic_embedding_to_d_model = nn.Linear(aspect_vector_key_size * aspect_vector_feature_count, self.d_model, bias=True).to(device)

    def forward(self, input_tensor, **kwargs):
        """
        :param input_tensor: the output of vanilla transformer embedding augmented with positional encoding information
        :param kwargs: must contain the bert indexed input token ids in a tensor named "bert_src"
        :return:
        """
        if "bert_src" in kwargs and kwargs["bert_src"] is not None:
            x_prime = self.linguistic_embedding_to_d_model(torch.cat(
                self.get_ling_embed_attention_keys_from_bert_converted_ids(kwargs["bert_src"]), dim=-1))
        else:
            raise ValueError("bert_src information are not provided!")
        return self.input_gate(torch.cat((x_prime, input_tensor), dim=-1))

    def get_ling_embed_attention_keys_from_bert_converted_ids(self, bert_input_sentences):
        input_ids = torch.tensor(bert_input_sentences, device=device)
        outputs = self.bert_lm(input_ids)[1]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                self.softmax(self.bert_weights_for_average_pooling))  # [:, 1:-1, :]
        # ##############################################################################################################
        # len(features_list) * batch_size * max_sequence_length, (H/ (len(features_list) + 1))
        keys = [hc(embedded).detach() for hc in self.aspect_vectors[:-1]]  # the last layer contains what we have not considered
        return keys


class AspectAugmentedTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(AspectAugmentedTransformer, self).__init__(SRC, TGT)
        d_model = int(cfg.transformer_d_model)
        self.integration_module = AspectIntegration(d_model)

    def init_model_params(self):
        super(AspectAugmentedTransformer, self).init_model_params()
        self.integration_module.init_model_params()

    def encode(self, input_tensor_with_lengths, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, _ = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        input_mask = self.generate_src_mask(input_tensor)
        x = self.integration_module(self.src_embed(input_tensor), **kwargs)
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask, input_tensor
