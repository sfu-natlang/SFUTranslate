"""
Implements a layer normalization module as explained in https://arxiv.org/abs/1607.06450. The module could be used to 
 force the output of model to follow the zero-mean unit-variance format. 
"""
from translate.backend.utils import backend

__author__ = "Hassan S. Shavarani"


class LayerNorm(backend.nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features: the size of the input and output of the layer
        :param eps: the value of epsilon to be added to the denominator of the normalizer to prevent division by zero.  
        """
        super(LayerNorm, self).__init__()
        self.a_2 = backend.nn.Parameter(backend.ones(features))
        self.b_2 = backend.nn.Parameter(backend.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
