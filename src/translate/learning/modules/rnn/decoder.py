"""
The RNN implementation of the Decoder module in the sequence to sequence framework plus the modular implementation of
 the attention module.
"""
from translate.backend.utils import backend, zeros_tensor
from translate.learning.modules.mlp.attention import GlobalAttention, LocalPredictiveAttention
from translate.logging.utils import logger

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
        attn = self.attn(backend.cat((decoder_input_tensor, decoder_hidden_layer), 1))[:, :encoder_output_tensors.size(0)]
        attn_weights = backend.nn.functional.softmax(attn, dim=1)
        attn_applied = backend.bmm(attn_weights.unsqueeze(1), encoder_output_tensors.transpose(0, 1)).transpose(0, 1)
        output = backend.cat((decoder_input_tensor, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)
        return output, attn_weights


class DecoderRNN(backend.nn.Module):
    def __init__(self, hidden_size: int, output_size: int, bidirectional_hidden_state: bool,
                 n_layers=1, batch_size=1, dropout_p=0.1, attention_method='dot', attention_type='global',
                 local_attention_d=0.0, padding_index=-1):
        """
        :param hidden_size: the output size of the last encoder hidden layer which is used as input in the decoder
        :param output_size: the output size of the decoder which is expected to be the size of the target vocabulary
        :param bidirectional_hidden_state: a flag indicating whether the encoder has been operating bidirectionally
        :param n_layers: number of expected decoder hidden layers
        :param batch_size: the expected size of batches passed through the decoder (note that this value might be
         different for some batches especially the last batches in the dataset)
        :param dropout_p: the dropout rate of the dropout layer applied to the input Embedding layer
        :param attention_method: the method of attention to be used, possible values ['dot', 'general', 'concat', 'add']
        :param attention_type: the type of attention to be either ['local' or 'global']
        :param local_attention_d: the diameter of attention span in case of performing the local attention
        :param padding_index: the index to be ignored for conversion to embedding vectors
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.encoder_is_bidirectional = bidirectional_hidden_state
        if bidirectional_hidden_state:
            self.hidden_size *= 2
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.num_directions = 1
        self.num_layers = n_layers
        self.batch_size = batch_size

        self.embedding = backend.nn.Embedding(self.output_size, self.hidden_size, padding_idx=padding_index)
        self.dropout = backend.nn.Dropout(self.dropout_p)
        self.lstm = backend.nn.LSTM(self.hidden_size * 2, self.hidden_size,  num_layers=n_layers)
        # self.out = backend.nn.Linear(self.hidden_size, self.output_size)
        # self.attention = Attention(self.hidden_size, self.max_length)
        if attention_type == 'local':
            logger.info("Loading the {} local-p attention with d={} ".format(attention_method, local_attention_d))
            self.attention = LocalPredictiveAttention(self.hidden_size, local_attention_d, method=attention_method)
        else:
            logger.info("Loading the global {} attention".format(attention_method))
            self.attention = GlobalAttention(self.hidden_size, method=attention_method)

    def get_hidden_size(self):
        return self.hidden_size

    def forward(self, input_tensor, hidden_layer_params, previous_h_hat, encoder_outputs, batch_size=-1):
        """
        :param input_tensor: : 1-D Tensor [batch_size]
        :param hidden_layer_params: Pair of size 2 of 2-D Tensors [batch_size, dec_hidden_size]
        :param previous_h_hat: 2-D Tensor [batch_size, decoder_hidden_size]
        :param encoder_outputs: 3-D Tensor [max_encoder_length, batch_size, encoder_hidden_size]
        :param batch_size: integer stating the size of :param input_tensor:
        :return: 2-D Tensor [batch_size, dec_hidden_size], Pair of size 2 of 2-D Tensors [batch_size, dec_hidden_size],
         2-D Tensor [batch_size, decoder_hidden_size], 2-D Tensor [batch_size, encoded_inputs_length]
        """
        if batch_size == -1:
            batch_size = self.batch_size
        embedded = self.embedding(input_tensor).view(1, batch_size, self.hidden_size)
        # embedded -> 2-D Tensor of size [batch_size, decoder_hidden_size]
        embedded = self.dropout(embedded)
        #  commented code: output, attn_weights = self.attention(encoder_outputs, embedded[0], hidden_layer[0])
        # output -> 3-D Tensor of size [1 (time), batch_size, decoder_hidden_size]
        output, hidden_layer_params = self.lstm(backend.cat((embedded, previous_h_hat.unsqueeze(0)), -1),
                                                hidden_layer_params)
        # h_hat -> 2-D Tensor [batch_size, decoder_hidden_size]
        # _ -> 2-D Tensor [batch_size, decoder_hidden_size]
        # attention_weights -> 2-D Tensor [batch_size, encoded_inputs_length]
        h_hat, _, attention_weights = self.attention(encoder_outputs, output[0])

        h_hat = backend.nn.functional.relu(h_hat)

        # commented code: output = backend.nn.functional.log_softmax(self.out(), dim=1)
        return h_hat, hidden_layer_params, attention_weights

    def reformat_encoder_hidden_states(self, encoder_hidden_prams):
        """
        :param encoder_hidden_prams: Pair of size 2 of 3-D Tensors [num_enc_dirs*n_enc_layers, batch_size, hidden_size]
        """
        hidden = encoder_hidden_prams[0]
        context = encoder_hidden_prams[1]
        if self.encoder_is_bidirectional:
            hidden = backend.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            context = backend.cat([context[0:context.size(0):2], context[1:context.size(0):2]], 2)
        if self.num_layers < hidden.size(0):
            hidden = hidden[hidden.size(0)-self.num_layers:]
            context = context[context.size(0)-self.num_layers:]
        return hidden, context

    def init_hidden(self, batch_size=-1):
        """
        The initializer for the hidden layer when starting to decode
        :param batch_size: expected batch size of the returning hidden state to be initialized
        """
        if batch_size == -1:
            batch_size = self.batch_size
        return zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size), \
            zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size)
