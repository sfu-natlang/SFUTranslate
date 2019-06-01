"""
The RNN implementation of the Encoder module in the sequence to sequence framework
"""
from translate.backend.utils import backend, zeros_tensor

__author__ = "Hassan S. Shavarani"


class EncoderRNN(backend.nn.Module):
    def __init__(self, input_size: int, emb_size:int, hidden_size: int, bidirectional: bool, n_layers: int = 1, batch_size: int = 1,
                 padding_index=-1):
        """
        :param input_size: the input size of the encoder which is expected to be the size of the source vocabulary
        :param hidden_size: the output size of the encoder hidden layer which is used as input in the decoder
        :param bidirectional: a flag indicating whether the encoder has been operating bidirectionally
        :param n_layers: number of expected encoder hidden layers
        :param batch_size: the expected size of batches passed through the encoder (note that this value might be
         different for some batches especially the last batches in the dataset)
        :param padding_index: the index to be ignored for conversion to embedding vectors
        """
        super(EncoderRNN, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_directions = 1
        if bidirectional:
            self.num_directions = 2
        self.num_layers = n_layers
        self.batch_size = batch_size
        self.embedding = backend.nn.Embedding(input_size, emb_size, padding_idx=padding_index)
        self.lstm = backend.nn.LSTM(emb_size, hidden_size, bidirectional=bidirectional, num_layers=n_layers)

    def forward(self, input_tensor, hidden_layer_params):
        """
        :param input_tensor: 2-D Tensor [max_length (might be different for each batch), batch_size]
        :param hidden_layer_params: Pair of size 2 of 3-D Tensors
          [num_enc_dirs*n_enc_layers, batch_size, hidden_size//n_enc_dirs]
        :return: 3-D Tensor [max_length, batch_size, hidden_size], hidden_layer_tensors same format as input
        """
        output = self.embedding(input_tensor)
        output, hidden_layer_params = self.lstm(output, hidden_layer_params)
        return output, hidden_layer_params

    def init_hidden(self, batch_size=-1):
        """
        The initializer for the hidden layer when starting to encode the input
        :param batch_size: expected batch size of the returning hidden state to be initialized
        """
        if batch_size == -1:
            batch_size = self.batch_size
        return zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size), \
            zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size)
