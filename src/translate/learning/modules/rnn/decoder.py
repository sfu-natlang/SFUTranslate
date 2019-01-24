"""
The RNN implementation of the Decoder module in the sequence to sequence framework plus the modular implementation of
 the attention module.
"""
from translate.backend.utils import backend, zeros_tensor

__author__ = "Hassan S. Shavarani"


class Attention(backend.nn.Module):
    def __init__(self, hidden_size: int, max_length: int):
        """
        :param hidden_size: the size of the hidden layer which is the input size of the decoder module.
          In case of bidirectional encoder this value is twice the actual value of the hidden layer size set in
           the encoder module
        :param max_length: the maximum expected length of a sequence which will be used to set the fixed size of
         the attention sub-modules
        """
        # TODO revisit the implementation with the actual paper formulae
        # TODO implement different attention models
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        # the linear layer in charge of converting the concatenation of the current decoder input and current decoder
        #  hidden state to attention scores. Replacement of the V_atanh(W_ah_t^{enc}+U_ah_{t'-1}^{dec}) in Loung et al.
        self.attn = backend.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = backend.nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, encoder_output_tensors, decoder_input_tensor, decoder_hidden_layer):
        attn = self.attn(backend.cat((decoder_input_tensor, decoder_hidden_layer), 1))
        attn_weights = backend.nn.functional.softmax(attn, dim=1)
        attn_applied = backend.bmm(attn_weights.unsqueeze(1), encoder_output_tensors.transpose(0, 1)).transpose(0, 1)
        output = backend.cat((decoder_input_tensor, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights


class DecoderRNN(backend.nn.Module):
    def __init__(self, hidden_size: int, output_size: int, bidirectional_hidden_state: bool, max_length: int,
                 n_layers=1, batch_size=1, dropout_p=0.1):
        """
        :param hidden_size: the output size of the last encoder hidden layer which is used as input in the decoder
        :param output_size: the output size of the decoder which is expected to be the size of the target vocabulary
        :param bidirectional_hidden_state: a flag indicating whether the encoder has been operating bidirectionally
        :param max_length: the maximum expected length of the input/output sequence
        :param n_layers: number of expected decoder hidden layers
        :param batch_size: the expected size of batches passed through the decoder (note that this value might be
         different for some batches especially the last batches in the dataset)
        :param dropout_p: the dropout rate of the dropout layer applied to the input Embedding layer
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        if bidirectional_hidden_state:
            self.hidden_size *= 2
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.num_directions = 1
        self.num_layers = n_layers
        self.batch_size = batch_size

        self.embedding = backend.nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = backend.nn.Dropout(self.dropout_p)
        self.lstm = backend.nn.LSTM(self.hidden_size, self.hidden_size)
        # self.out = backend.nn.Linear(self.hidden_size, self.output_size)
        self.attention = Attention(self.hidden_size, self.max_length)

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, input_tensor, hidden_layer, context, encoder_outputs, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        embedded = self.embedding(input_tensor).view(1, batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        output, attn_weights = self.attention(encoder_outputs, embedded[0], hidden_layer[0])

        output = backend.nn.functional.relu(output)
        output, (hidden_layer, context) = self.lstm(output, (hidden_layer, context))

        # output = backend.nn.functional.log_softmax(self.out(), dim=1)
        return output[0], (hidden_layer, context), attn_weights

    def init_hidden(self, batch_size=-1):
        """
        The initializer for the hidden layer when starting to decode
        :param batch_size: expected batch size of the returning hidden state to be initialized
        """
        if batch_size == -1:
            batch_size = self.batch_size
        return zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size), \
            zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size)
