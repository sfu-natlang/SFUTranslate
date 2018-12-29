"""
The implementation of the Encoder module and the EncoderLayers used in the Encoder module as explained in the "attention
 is all you need" paper.
"""
from translate.backend.utils import backend
from translate.learning.modules.mlp.layernorm import LayerNorm
from translate.learning.modules.transformer.utils import clones, SublayerConnection

__author__ = "Hassan S. Shavarani"


class Encoder(backend.nn.Module):
    def __init__(self, layer, N):
        """
        Implementation of the encoder as a stack of :param N: :param layer: instances
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn and return the normalized values of the output
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(backend.nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        Each encoder layer is made of a :param self_attn: sub-layer, and a :param feed_forward: sub-layer which are 
         applied consecutively to the input to create the output tensor. For information on how each work you may want 
          to look at :class MultiHeadedAttention: and :class PositionwiseFeedForward: implementations. 
           :param size: is the layer-normalization layer size applied to the output and :param dropout: is the dropout 
            probability of the output layer before returning the output tensor. 
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
