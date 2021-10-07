"""
This file contains the implementation of different extensions of vanilla Transformer model. The extensions particularly are either models that use
    aspect vectors (aspect-augmented; hence being in "aspects" package) or are the baselines used to be compared with aspect-augmented models.
    The following models are implemented here:
    - AspectAugmentedTransformer: the implementation of aspect-augmented NMT model by Shavarani and Sarkar (2020)
        "Better Neural Machine Translation by Extracting Linguistic Information from BERT"
    - MultiHeadAspectAugmentedTransformer: the implementation of aspect-augmented NMT model which replaces the MLP aspect integration module with a
        multi-head attention module over different token aspects
    - SyntaxInfusedTransformer: the implementation of syntax-infused NMT model by Sundararaman et al. (2019) [https://arxiv.org/abs/1911.06156]
        "Syntax-Infused Transformer and BERT models for Machine Translation and Natural Language Understanding"
    - BertFreezeTransformer: the implementation of bert-freeze input feeding to Transformer by  Clinchant et al. (2019) [https://www.aclweb.org/anthology/D19-5611.pdf]
        "On the use of BERT for Neural Machine Translation"
        Our implementation does not just initialize the encoder embedding layer with bert parameters but rather actively embeds each sentence with
            bert and feeds the acquired bert embeddings as input to the encoder layers of Transformer.
"""
import torchtext
if torchtext.__version__.startswith('0.9') or torchtext.__version__.startswith('0.10'):
    from torchtext.legacy import data
else:
    from torchtext import data

from configuration import cfg, device
from models.aspects.module import AspectIntegration, MultiHeadAspectAugmentationLayer, SyntaxInfusedSRCEmbedding, BertEmbeddingIntegration
from models.transformer.model import Transformer
from models.transformer.modules import Embeddings, PositionalEncoding


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


class MultiHeadAspectAugmentedTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field, integrate_before_pe=False, use_left_over_vector=False, value_from_token_embedding=False):
        super(MultiHeadAspectAugmentedTransformer, self).__init__(SRC, TGT)
        self.multi_head_aspect_integration_layer = MultiHeadAspectAugmentationLayer(use_left_over_vector, value_from_token_embedding)
        self.integrate_before_pe = integrate_before_pe
        print("Aspect multi-head attention will be integrated to token embeddings {} applying the positional embeddings".format(
            "before" if integrate_before_pe else "after"))
        if self.integrate_before_pe:
            self.src_embed = Embeddings(self.multi_head_aspect_integration_layer.d_model, len(SRC.vocab))
            dropout = float(cfg.transformer_dropout)
            max_len = int(cfg.transformer_max_len)
            self.positional_encoding = PositionalEncoding(self.multi_head_aspect_integration_layer.d_model, dropout, max_len)
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
        if self.integrate_before_pe:
            x = self.positional_encoding(x)
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask, input_tensor


class SyntaxInfusedTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(SyntaxInfusedTransformer, self).__init__(SRC, TGT)
        # TODO features_dict from SyntaxInfusedInformationContainer for test mode
        self.src_embed = SyntaxInfusedSRCEmbedding(len(SRC.vocab)).to(device)

    def init_model_params(self):
        super(SyntaxInfusedTransformer, self).init_model_params()

    def encode(self, input_tensor_with_lengths, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, _ = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        input_mask = self.generate_src_mask(input_tensor)
        x = self.src_embed(input_tensor, **kwargs)
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask, input_tensor


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
