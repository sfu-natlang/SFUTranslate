"""
The RNN implementation of the Generator module in the sequence to sequence framework which maps back the embedded hidden 
 state passed to it into target vocabulary space
"""
from translate.models.backend.utils import backend

__author__ = "Hassan S. Shavarani"


class GeneratorNN(backend.nn.Module):
    def __init__(self, hidden_size: int, output_size: int, dropout_p=0.1):
        """
        :param hidden_size: the output size of the last encoder hidden layer which is used as input in the decoder
        :param output_size: the output size of the decoder which is expected to be the size of the target vocabulary
        :param dropout_p: the dropout rate of the dropout layer applied to the input Embedding layer
        """
        super(GeneratorNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.dropout = backend.nn.Dropout(self.dropout_p)

        self.out = backend.nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor):
        """
        :param input_tensor: the embedded batched vector to be mapped back to the vocabulary space 
        """
        return backend.nn.functional.log_softmax(self.out(input_tensor), dim=1)
