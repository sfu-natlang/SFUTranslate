"""
The implementation of Bert Freeze model which changes the embedding layer of the vanilla transformer model to a pre-trained frozen Bert model
"""
import torch
from torch import nn
from torchtext import data

from configuration import cfg, device
from models.transformer.model import Transformer
from models.transformer.modules import PositionalEncoding
from readers.data_provider import src_tokenizer_obj

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    from transformers import BertForMaskedLM
except ImportError:
    warnings.warn("transformers package is not available, transformers.BertForMaskedLM will not be accessible.")
    BertForMaskedLM = None


class BertEmbeddingIntegration(nn.Module):
    def __init__(self, d_model):
        super(BertEmbeddingIntegration, self).__init__()
        self.d_model = d_model
        self.bert_lm = None
        self.softmax = nn.Softmax(dim=-1)
        self.bert_weights_for_average_pooling = None
        self.number_of_bert_layers = 0
        self.output_bridge = None

    def init_model_params(self):
        print("Running the init params for BERT language model")
        self.bert_lm = BertForMaskedLM.from_pretrained(src_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        self.number_of_bert_layers = len(self.bert_lm.bert.encoder.layer) + 1
        self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers), requires_grad=True).to(device)
        for p in self.bert_weights_for_average_pooling:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # TODO if final output size if not equal to self.d_model convert it back to self.d_model space using self.output_bridge
        lm_hidden_size = self.bert_lm.bert.pooler.dense.in_features
        if self.d_model != lm_hidden_size:
            self.output_bridge = nn.Linear(lm_hidden_size, self.d_model).to(device)
            for p in self.output_bridge.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, **kwargs):
        """
        :param kwargs: must contain the bert indexed input token ids in a tensor named "bert_src"
        :return:
        """
        if "bert_src" in kwargs and kwargs["bert_src"] is not None:
            return self.get_ling_embed_attention_keys_from_bert_converted_ids(kwargs["bert_src"])
        else:
            raise ValueError("bert_src information are not provided!")

    def get_ling_embed_attention_keys_from_bert_converted_ids(self, bert_input_sentences):
        input_ids = torch.tensor(bert_input_sentences, device=device)
        outputs = self.bert_lm(input_ids)[1]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
        # TODO "self.bert_weights_for_average_pooling.to(device)" should not be done each time.
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0), self.softmax(self.bert_weights_for_average_pooling.to(device)))  # [:, 1:-1, :]
        if self.output_bridge is not None:
            embedded = self.output_bridge(embedded)
        return embedded


class BertFreezeTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(BertFreezeTransformer, self).__init__(SRC, TGT)
        d_model = int(cfg.transformer_d_model)
        dropout = float(cfg.transformer_dropout)
        max_len = int(cfg.transformer_max_len)
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len).to(device)
        self.src_embed = BertEmbeddingIntegration(d_model)

    def init_model_params(self):
        super(BertFreezeTransformer, self).init_model_params()
        self.src_embed.init_model_params()

    def encode(self, input_tensor_with_lengths, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, _ = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        input_mask = self.generate_src_mask(input_tensor)
        x = self.positional_encoding(self.src_embed(**kwargs))
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask, input_tensor
