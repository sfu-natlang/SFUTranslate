"""
The implementation of Multi-Head Aspect Augmented Transformer model along with the modified multi-head attention module
"""
import torch
from torch import nn
from torchtext import data

from configuration import cfg, device
from models.transformer.model import Transformer
from readers.data_provider import src_tokenizer_obj
from models.transformer.utils import clones, attention
from models.transformer.modules import SublayerConnection

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import BertForMaskedLM
except ImportError:
    warnings.warn("transformers package is not available, transformers.BertForMaskedLM will not be accessible.")
    BertForMaskedLM = None


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_aspects, dropout=0.1):
        """
        Implements Figure 2 (right) of the paper (https://arxiv.org/pdf/1706.03762.pdf)
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.linears_aspects = clones(nn.Linear(d_aspects, self.d_k), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, aspect_vectors_list, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        key = value = torch.cat([e.unsqueeze(1) for e in aspect_vectors_list], dim=1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query,))]
        key, value = [l(x) for l, x in zip(self.linears_aspects, (key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadAspectAugmentationLayer(nn.Module):
    """
    Encoder is made up of self-attn and feed forward
    """
    def __init__(self):
        super(MultiHeadAspectAugmentationLayer, self).__init__()
        self.d_model = int(cfg.transformer_d_model)
        dropout = float(cfg.transformer_dropout)
        self.sublayer = clones(SublayerConnection(self.d_model, dropout), 2)
        self.aspect_attn = None
        self.bert_lm = None
        self.aspect_vectors = None
        self.bert_weights_for_average_pooling = None
        self.softmax = nn.Softmax(dim=-1)

    def init_model_params(self):
        print("Running the init params for BERT language model")
        self.bert_lm = BertForMaskedLM.from_pretrained(src_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        # self.number_of_bert_layers = len(self.bert_lm.bert.encoder.layer) + 1
        # self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers), requires_grad=True)
        print("Running the init params for aspect_vectors")
        try:
            so = torch.load(cfg.aspect_vectors_data_address, map_location=lambda storage, loc: storage)
        except KeyError:
            print("MultiHeadAspectAugmentationLayer failed to initialize since \"aspect_vectors_data_address\" is not set in the config file.\n"
                  "Exiting!")
            exit()
        self.aspect_vectors = so['aspect_vectors'].to(device)
        self.bert_weights_for_average_pooling = nn.Parameter(so['bert_weights'].to(device), requires_grad=True)
        aspect_vector_key_size = self.aspect_vectors[0].out_features
        aspect_vector_feature_count = len(self.aspect_vectors) - 1
        self.aspect_attn = MultiHeadedAttention(aspect_vector_feature_count, self.d_model, aspect_vector_key_size).to(device)

    def forward(self, x, mask, **kwargs):
        """
        Follow Figure 1 (left) for connections [https://arxiv.org/pdf/1706.03762.pdf]
        """
        if "bert_src" in kwargs and kwargs["bert_src"] is not None:
            x_prime = self.get_ling_embed_attention_keys_from_bert_converted_ids(kwargs["bert_src"])
        else:
            raise ValueError("bert_src information are not provided!")
        return self.sublayer[0](x, lambda x: self.aspect_attn(x, x_prime, mask))

    def get_ling_embed_attention_keys_from_bert_converted_ids(self, bert_input_sentences):
        input_ids = torch.tensor(bert_input_sentences, device=device)
        outputs = self.bert_lm(input_ids)[1]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                self.softmax(self.bert_weights_for_average_pooling))  # [:, 1:-1, :]
        keys = [hc(embedded).detach() for hc in self.aspect_vectors[:-1]]  # the last layer contains what we have not considered
        return keys


class MultiHeadAspectAugmentedTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(MultiHeadAspectAugmentedTransformer, self).__init__(SRC, TGT)
        self.multi_head_aspect_integration_layer = MultiHeadAspectAugmentationLayer()
        print("Multi-head aspect augmented transformer model created ...")

    def init_model_params(self):
        super(MultiHeadAspectAugmentedTransformer, self).init_model_params()
        self.multi_head_aspect_integration_layer.init_model_params()

    def encode(self, input_tensor_with_lengths, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, _ = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        input_mask = self.generate_src_mask(input_tensor)
        x = self.src_embed(input_tensor)
        x = self.multi_head_aspect_integration_layer(x, input_mask, **kwargs)
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask, input_tensor
