"""
Implementation of the multi-head attention layer as described in the "attention is all you need" paper.
"""
import math
from translate.backend.utils import backend
from translate.learning.modules.transformer.utils import clones

__author__ = "Hassan S. Shavarani"


class MultiHeadedAttention(backend.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        :param h: the number of heads 
        :param d_model: the size of the input to the layer
        :param dropout: the dropout probability of the attention layer while producing the output 
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(backend.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = backend.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        """
        Computes the 'Scaled Dot Product Attention' as detailed in the "attention is all you need" paper.
        """
        d_k = query.size(-1)
        scores = backend.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = backend.nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return backend.matmul(p_attn, value), p_attn
