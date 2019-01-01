"""
Implements a feed forward 2 layer perceptron mainly implemented to be used as a part of transformer architecture
"""
from translate.backend.utils import backend

__author__ = "Hassan S. Shavarani"


class PositionwiseFeedForward(backend.nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        :param d_model: the input size and output size of the module 
        :param d_ff: the hidden layer size of the module
        :param dropout: the dropout probability applied to the output of the hidden layer before getting forwarded 
         to the next layer 
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = backend.nn.Linear(d_model, d_ff)
        self.w_2 = backend.nn.Linear(d_ff, d_model)
        self.dropout = backend.nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(backend.nn.functional.relu(self.w_1(x))))
