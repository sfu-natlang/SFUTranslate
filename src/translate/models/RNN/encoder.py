"""
The RNN implementation of the Encoder module in the sequence to sequence framework
"""
from translate.models.backend.utils import backend, zeros_tensor

__author__ = "Hassan S. Shavarani"


class EncoderRNN(backend.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bidirectional: bool, n_layers: int = 1, batch_size: int = 1):
        """
        :param input_size: the input size of the encoder which is expected to be the size of the source vocabulary
        :param hidden_size: the output size of the encoder hidden layer which is used as input in the decoder
        :param bidirectional: a flag indicating whether the encoder has been operating bidirectionally
        :param n_layers: number of expected encoder hidden layers
        :param batch_size: the expected size of batches passed through the encoder (note that this value might be
         different for some batches especially the last batches in the dataset)
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_directions = 1
        if bidirectional:
            self.num_directions = 2
        self.num_layers = n_layers
        self.batch_size = batch_size
        self.embedding = backend.nn.Embedding(input_size, hidden_size)
        self.gru = backend.nn.GRU(hidden_size, hidden_size, bidirectional=bidirectional, num_layers=n_layers)

    def forward(self, input_tensor, hidden_layer, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        output = self.embedding(input_tensor).view(1, batch_size, self.hidden_size)
        output, hidden_layer = self.gru(output, hidden_layer)
        return output, hidden_layer

    def init_hidden(self, batch_size=-1):
        """
        The initializer for the hidden layer when starting to encode the input
        :param batch_size: expected batch size of the returning hidden state to be initialized
        """
        if batch_size == -1:
            batch_size = self.batch_size
        return zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size)
