"""
The modular implementation of the Attention modules in the sequence to sequence framework.
"""
from translate.backend.utils import backend, zeros_tensor, device, list_to_long_tensor

__author__ = "Hassan S. Shavarani"


class GlobalAttention(backend.nn.Module):
    def __init__(self, hidden_size, method='concat'):
        """
        the linear layer in charge of converting the concatenation of the current decoder input and current decoder
         hidden state to attention scores.
        :param hidden_size: the input size of the decoder module.
        :param method: the method of attention az mentioned in Loung et al., 2015. or Bahdanau et al., 2015
         Possible Values:['dot' | 'general' | 'concat' | 'additive']
        """
        super(GlobalAttention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.use_cuda = backend.cuda.is_available()

        if self.method == 'general':
            self.w = backend.nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        elif self.method == 'concat':
            self.w = backend.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
            self.v = backend.nn.Linear(self.hidden_size, 1, bias=False)
        elif self.method == 'add':
            self.w = backend.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.u = backend.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.v = backend.nn.Linear(self.hidden_size, 1, bias=False)

        self.out = backend.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, encoder_output_tensors, decoder_hidden_layer):
        """
        :param encoder_output_tensors: 3-D Tensor [max_length, batch_size, hidden_layer_size]
        :param decoder_hidden_layer: 1-D Tensor [hidden_layer_size]
        :return: 2-D Tensor of size [batch_size, hidden_layer_size],
                 2-D Tensor of size [batch_size, hidden_layer_size],
                 2-D Tensor of size [batch_size, max_length]
        """
        seq_len = encoder_output_tensors.size(0)
        batch_size = encoder_output_tensors.size(1)
        hidden_size = encoder_output_tensors.size(2)

        # Create variable to store attention energies of size [max_length, batch_size]
        attn_energies = zeros_tensor(seq_len, batch_size, 1).squeeze(-1).to(device)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(decoder_hidden_layer, encoder_output_tensors[i], batch_size, hidden_size)

        attention_weights = backend.nn.functional.softmax(attn_energies.transpose(0, 1), dim=1)
        context_vector = attention_weights.view(batch_size, 1, seq_len).bmm(
            encoder_output_tensors.transpose(0, 1)).view(batch_size, hidden_size)

        h_hat = backend.cat((decoder_hidden_layer, context_vector), 1)
        # Equation (5) in Loung et al. 2015 implies that the following transformation should be applied to the h_hat
        h_hat = backend.tanh(self.out(h_hat))
        return h_hat, context_vector, attention_weights

    def score(self, hidden, encoder_output, batch_size, hidden_size):
        """
        :param hidden: 2-D tensor of size [batch_size, decoder_hidden_size]
        :param encoder_output: 2-D tensor of size [batch_size, decoder_hidden_size]
        :param batch_size: the first dimension of the passed tensors
        :param hidden_size: the second dimension of the passed tensors
        :return: 1-D Tensor of size [batch_size]
        """
        if self.method == 'dot':
            energy = hidden.view(batch_size, 1, hidden_size).bmm(
                encoder_output.view(batch_size, hidden_size, 1)).view(-1)
            return energy

        elif self.method == 'general':
            energy = hidden.view(batch_size, 1, hidden_size).bmm(
                self.w(encoder_output).view(batch_size, hidden_size, 1)).view(-1)
            return energy

        elif self.method == 'concat':
            energy = self.v(self.w(backend.cat((hidden, encoder_output), 1))).view(-1)
            return energy

        elif self.method == 'add':
            energy = self.v(self.w(hidden) + self.u(encoder_output)).view(-1)
            return energy


class LocalPredictiveAttention(GlobalAttention):
    def __init__(self, hidden_size, d, method='concat'):
        """
        the linear layer in charge of converting the concatenation of the current decoder input and current decoder
         hidden state to attention scores.
        :param hidden_size: the input size of the decoder module.
        :param d: window span size for attention
        :param method: the method of attention az mentioned in Loung et al., 2015. or Bahdanau et al., 2015
         Possible Values:['dot' | 'general' | 'concat' | 'additive']
        """
        super(LocalPredictiveAttention, self).__init__(hidden_size, method)
        self.d = float(d)
        self.w_p = backend.nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_p = backend.nn.Linear(self.hidden_size, 1, bias=False)

    def forward(self, encoder_output_tensors, decoder_hidden_layer):
        """
        :param encoder_output_tensors: 3-D Tensor [max_length, batch_size, hidden_layer_size]
        :param decoder_hidden_layer: 1-D Tensor [hidden_layer_size]
        :return: 2-D Tensor of size [batch_size, hidden_layer_size],
                 2-D Tensor of size [batch_size, hidden_layer_size],
                 2-D Tensor of size [batch_size, max_length]
        """
        seq_len = encoder_output_tensors.size(0)
        batch_size = encoder_output_tensors.size(1)
        hidden_size = encoder_output_tensors.size(2)

        # an integer pointing to one of the locations of the sentence
        p_t = seq_len * backend.sigmoid(self.v_p(backend.tanh(self.w_p(decoder_hidden_layer))))
        # 1-D Tensor pf size [batch_size, seq_len]
        local_scores = backend.exp(-((list_to_long_tensor(range(seq_len)).float().to(device) - p_t) ** 2) / (
            2 * (self.d / 2) ** 2))

        # Create variable to store attention energies of size [max_length, batch_size]
        attn_energies = zeros_tensor(seq_len, batch_size, 1).squeeze(-1).to(device)

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(decoder_hidden_layer, encoder_output_tensors[i], batch_size, hidden_size)

        # 2-D Tensor of size [batch_size, max_length]
        attention_weights = backend.nn.functional.softmax(attn_energies.transpose(0, 1), dim=1) * local_scores

        # could potentially need to be replaced with the following line
        # attention_weights = backend.nn.functional.softmax(attention_weights * local_scores)

        context_vector = attention_weights.view(batch_size, 1, seq_len).bmm(
            encoder_output_tensors.transpose(0, 1)).view(batch_size, hidden_size)

        h_hat = backend.cat((decoder_hidden_layer, context_vector), 1)
        # Equation (5) in Loung et al. 2015 implies that the following transformation should be applied to the h_hat
        h_hat = backend.tanh(self.out(h_hat))
        return h_hat, context_vector, attention_weights
