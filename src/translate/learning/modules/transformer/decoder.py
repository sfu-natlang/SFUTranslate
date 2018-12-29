"""
The implementation of the Decoder module and the DecoderLayers used in the Decoder module as explained in the "attention
 is all you need" paper.
"""

from translate.backend.utils import backend
from translate.learning.modules.mlp.layernorm import LayerNorm
from translate.learning.modules.transformer.utils import clones, SublayerConnection

__author__ = "Hassan S. Shavarani"


class Decoder(backend.nn.Module):
    """
    Generic N layer decoder with masking
    """

    def __init__(self, layer, N):
        """
        the Decoder is made of :param N: copies of the :param layer: instance passed to it.   
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(backend.nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        Each decoder layer is made of a :param self_attn: sub-layer, :param src_attn: sub-layer, and a 
         :param feed_forward: sub-layer which are applied consecutively to the input to create the output tensor. 
          For information on how each work you may want to look at :class MultiHeadedAttention: and 
           :class PositionwiseFeedForward: implementations. :param size: is the layer-normalization layer size applied 
            to the output and :param dropout: is the dropout probability of the output layer before returning the output 
             tensor. 
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
