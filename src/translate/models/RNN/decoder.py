from translate.models.backend.utils import backend, zeros_tensor


class Attention(backend.nn.Module):
    def __init__(self, hidden_size: int, max_length: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.attn = backend.nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = backend.nn.Linear(self.hidden_size * 2, self.hidden_size)

    def forward(self, encoder_output_tensors, decoder_input_tensor, decoder_hidden_layer):
        attn = self.attn(backend.cat((decoder_input_tensor, decoder_hidden_layer), 1))
        attn_weights = backend.nn.functional.softmax(attn, dim=1)
        attn_applied = backend.bmm(attn_weights.unsqueeze(1), encoder_output_tensors.transpose(0, 1)).transpose(0, 1)
        return attn_applied, attn_weights


class DecoderRNN(backend.nn.Module):
    def __init__(self, hidden_size: int, output_size: int, bidirectional_hidden_state: bool, max_length: int,
                 n_layers=1, batch_size=1, dropout_p=0.1):
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
        self.gru = backend.nn.GRU(self.hidden_size, self.hidden_size)
        self.out = backend.nn.Linear(self.hidden_size, self.output_size)
        self.attention = Attention(self.hidden_size, self.max_length)

    def forward(self, input_tensor, hidden_layer, encoder_outputs, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        embedded = self.embedding(input_tensor).view(1, batch_size, self.hidden_size)
        embedded = self.dropout(embedded)

        attn_applied, attn_weights = self.attention(encoder_outputs, embedded[0], hidden_layer[0])

        output = backend.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = backend.nn.functional.relu(output)
        output, hidden_layer = self.gru(output, hidden_layer)

        output = backend.nn.functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden_layer, attn_weights

    def init_hidden(self, batch_size=-1):
        if batch_size == -1:
            batch_size = self.batch_size
        return zeros_tensor(self.num_directions * self.num_layers, batch_size, self.hidden_size)
