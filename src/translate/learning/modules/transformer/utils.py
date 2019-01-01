"""
Implements all the necessary utility functions for the transformer module.
"""
import copy, math
from translate.backend.utils import backend, Variable
from translate.learning.modules.mlp.layernorm import LayerNorm

__author__ = "Hassan S. Shavarani"


def clones(module_instance, N):
    """Produce N identical layers."""
    return backend.nn.ModuleList([copy.deepcopy(module_instance) for _ in range(N)])


class SublayerConnection(backend.nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = backend.nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class Embeddings(backend.nn.Module):
    """
    The embedding layer values as detailed in the "attention is all you need paper"
    """

    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = backend.nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(backend.nn.Module):
    """
    Implements the sinusoidal positional embedding layer as detailed in the "attention is all you need" paper.
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = backend.nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        pe = backend.zeros(max_len, d_model)
        position = backend.arange(0., max_len).unsqueeze(1)
        div_term = backend.exp(backend.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = backend.sin(position * div_term)
        pe[:, 1::2] = backend.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)
