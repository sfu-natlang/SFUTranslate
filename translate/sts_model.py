import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchtext import data
from configuration import cfg, device


class STS(nn.Module):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(STS, self).__init__()
        self.SRC = SRC
        self.TGT = TGT
        self.criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[cfg.pad_token], reduction='sum')
        self.bahdanau_attention = bool(cfg.bahdanau_attention)
        print("Creating the Seq2Seq Model with {} attention".format("Bahdanau" if self.bahdanau_attention else "Loung"))
        self.coverage = bool(cfg.coverage_required)
        print("Coverage model (linguistic definition) is {}".format(
            "also considered" if self.coverage else "not considered"))
        self.encoder_emb = nn.Embedding(len(SRC.vocab), int(cfg.encoder_emb_size),
                                        padding_idx=SRC.vocab.stoi[cfg.pad_token])
        self.decoder_emb = nn.Embedding(len(TGT.vocab), int(cfg.decoder_emb_size),
                                        padding_idx=TGT.vocab.stoi[cfg.pad_token])
        self.encoder_layers = int(cfg.encoder_layers)
        self.encoder_bidirectional = True
        self.encoder_hidden = int(cfg.encoder_hidden_size)
        self.encoder = nn.LSTM(int(cfg.encoder_emb_size), self.encoder_hidden, self.encoder_layers,
                               bidirectional=self.encoder_bidirectional,
                               dropout=float(cfg.encoder_dropout_rate) if self.encoder_layers > 1 else 0.0)
        self.decoder_hidden = int(cfg.decoder_hidden_size)
        self.attention_W = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1),
                                     self.decoder_hidden, bias=False)
        if self.bahdanau_attention:
            self.attention_U = nn.Linear(self.decoder_hidden, self.decoder_hidden, bias=False)
            self.attention_V = nn.Linear(self.decoder_hidden, 1, bias=False)
        else:
            self.attention_U = None
            self.attention_V = None
        # self.attention_proj = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1)
        #                                 + self.decoder_hidden, self.decoder_hidden, bias=False)
        if self.encoder_hidden * (2 if self.encoder_bidirectional else 1) != self.decoder_hidden:
            self.enc_dec_hidden_bridge = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1),
                                                   self.decoder_hidden, bias=False)
        else:
            self.enc_dec_hidden_bridge = None
        self.decoder_input_size = int(cfg.decoder_emb_size) + self.encoder_hidden * (
            2 if self.encoder_bidirectional else 1)
        self.decoder_layers = int(cfg.decoder_layers)
        self.decoder = nn.LSTM(self.decoder_input_size, self.decoder_hidden, self.decoder_layers,
                               dropout=float(cfg.decoder_dropout_rate) if self.decoder_layers > 1 else 0.0)
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.emb_dropout = nn.Dropout(p=float(cfg.emb_dropout))
        self.out_dropout = nn.Dropout(p=float(cfg.out_dropout))
        self.out = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1)
                             + self.decoder_hidden + int(cfg.decoder_emb_size), len(TGT.vocab))
        if self.coverage:
            if not self.bahdanau_attention:
                raise ValueError("Coverage model is just integrated with Bahdanau Attention")
            self.u_phi_j = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1), 1, bias=False)
            self.phi_j_N = int(cfg.coverage_phi_n)
            self.coverage_lambda = float(cfg.coverage_lambda)
            self.attention_C = nn.Linear(1, self.decoder_hidden, bias=False)
            self.coverage_dropout = nn.Dropout(p=float(cfg.coverage_dropout))
        else:
            self.u_phi_j = None
            self.phi_j_N = 1
            self.coverage_lambda = 0.0
            self.attention_C = None
            self.coverage_dropout = None

    def forward(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, input_lengths = input_tensor_with_lengths
        if output_tensor_with_length is not None:
            output_tensor, outputs_lengths = output_tensor_with_length
            tokens_count = float(outputs_lengths.sum().item())
        else:
            output_tensor, outputs_lengths = None, None
            tokens_count = 0.0
        predicted_tokens_count = 0.0
        input_sequence_length, batch_size = input_tensor.size()
        embedded_input = self.emb_dropout(self.encoder_emb(input_tensor))  # seq_length * batch_size * emd_size
        packed_input = pack_padded_sequence(embedded_input, input_lengths, enforce_sorted=False)
        encoded_pack_output, encoder_lstm_context = self.encoder(packed_input)
        encoder_lstm_output, _ = pad_packed_sequence(encoded_pack_output)
        # encoder_lstm_output, encoder_lstm_context = self.encoder(
        # embedded_input, self.encoder_init(input_tensor.size(1)))
        decoder_lstm_context = self.reformat_encoder_hidden_states(encoder_lstm_context)
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        next_token = output_tensor.select(0, 0) if output_tensor is not None and not test_mode else \
            torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.bos_token]).to(device)  # size = batch_size
        pad_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.pad_token]).to(device)
        result = torch.zeros(target_length, batch_size, device=device)
        cumulative_loss = 0.0
        c_t = torch.zeros(batch_size, self.encoder_hidden * (2 if self.encoder_bidirectional else 1), device=device)
        eos_predicted = torch.zeros(batch_size, device=device).byte()
        if self.bahdanau_attention:
            preprocessed_attention_encoder_representations = self.attention_W(encoder_lstm_output).transpose(0, 1)
        else:
            preprocessed_attention_encoder_representations = self.attention_W(encoder_lstm_output).permute(1, 2, 0)
        attention_mask = input_tensor.transpose(0, 1).unsqueeze(1) != self.SRC.vocab.stoi[cfg.pad_token]
        loss_size = 0.0

        if self.coverage:
            coverage_vector = torch.zeros(batch_size, input_sequence_length, 1, device=device).float()
            # phi_j: batch_size * 1 * max_input_length
            phi_j = self.phi_j_N * self.sigmoid(
                self.u_phi_j(self.coverage_dropout(encoder_lstm_output.transpose(0, 1)))).squeeze(-1).unsqueeze(1)
        max_attention_indices = torch.zeros(target_length, batch_size, device=device)
        for t in range(target_length):
            dec_emb = self.emb_dropout(self.decoder_emb(next_token))  # batch_size * decoder_emb_size
            decoder_input = torch.cat([dec_emb, c_t], dim=1).view(1, batch_size, self.decoder_input_size)
            _, decoder_lstm_context = self.decoder(decoder_input, decoder_lstm_context)
            query = decoder_lstm_context[0][-1].view(batch_size, self.decoder_hidden)
            semi_output = self.out_dropout(torch.cat([dec_emb, query, c_t], dim=1))
            o = self.out(self.tanh(semi_output)) \
                .view(batch_size, len(self.TGT.vocab))  # batch_size, target_vocab_size
            greedy_prediction = torch.argmax(o, dim=1).detach()
            eos_predicted = torch.max(eos_predicted, (greedy_prediction == self.TGT.vocab.stoi[cfg.eos_token]))
            if output_tensor is not None:
                # Input Feeding
                next_token = output_tensor.select(0, t + 1) if t < output_tensor.size(0) - 1 else pad_token
                cumulative_loss += self.criterion(o, next_token)
                loss_size += 1.0
            else:
                # greedy approach
                next_token = greedy_prediction
            predicted_tokens_count += batch_size - eos_predicted.sum().item()
            if sum(eos_predicted.int()) == batch_size:
                break
            # overwrite the Input Feeding criteria
            if test_mode:
                next_token = greedy_prediction
            result[t, :] = greedy_prediction
            # Calculate the new context vector
            # encoded representations size: innput_seq_length * batch_size * (2 * encoder_hidden representation size)
            # decoder representation size: 1 * batch_size * decoder representation size
            # inp_hidden = encoder_lstm_output.transpose(0, 1)
            # mapped_inputs = self.attention_W(inp_hidden)  # batch_size, input_len, dec hidden
            # alphas = torch.bmm(query.unsqueeze(1), mapped_inputs.transpose(1, 2))  # batch_size,1, input_len
            # alphas = self.softmax(alphas)
            # c_t = torch.bmm(alphas, inp_hidden).squeeze(1)
            if self.bahdanau_attention:
                attention_inputs = self.attention_U(query.unsqueeze(1).repeat(1, input_sequence_length, 1)) + \
                                   preprocessed_attention_encoder_representations
                if self.coverage:
                    attention_inputs = attention_inputs + self.attention_C(coverage_vector)
                alphas = self.attention_V(self.tanh(attention_inputs)).squeeze(2).unsqueeze(1) # b_size,1,input_len
            else:  # Loung general
                alphas = query.unsqueeze(1) @ preprocessed_attention_encoder_representations  # b_size,1,input_len
            alphas = torch.where(attention_mask, alphas, alphas.new_full([1], float('-inf')))
            alphas = self.softmax(alphas)  # batch_size * 1 * max_input_length
            if self.coverage:
                coverage_vector = coverage_vector + ((1.0 / (phi_j + 1e-32)) * alphas).squeeze(1).unsqueeze(-1)
            # input_tensor => max_seq_length * batch_size
            max_attention_indices[t, :] = alphas.max(dim=-1)[1].view(-1).detach()  # batch_size
            c_t = (alphas @ encoder_lstm_output.transpose(0, 1)).squeeze(1)
        if self.coverage and output_tensor is not None:
            expected_coverage = torch.ones(batch_size, input_sequence_length, device=device).float()
            cumulative_loss = cumulative_loss + self.coverage_lambda * torch.pow(
                expected_coverage - coverage_vector.squeeze(2) * attention_mask.float().squeeze(1), 2).sum()
        return result, max_attention_indices, cumulative_loss,  loss_size, tokens_count

    def reformat_encoder_hidden_states(self, encoder_hidden_prams):
        """
        :param encoder_hidden_prams: Pair of size 2 of 3-D Tensors
        [num_encoder_directions*num_encoder_layers, batch_size, hidden_size]
        """
        hidden = encoder_hidden_prams[0]
        context = encoder_hidden_prams[1]
        if self.encoder_bidirectional:
            hidden = torch.cat([hidden[0:hidden.size(0):2], hidden[1:hidden.size(0):2]], 2)
            context = torch.cat([context[0:context.size(0):2], context[1:context.size(0):2]], 2)
        if self.decoder_layers < hidden.size(0):
            hidden = hidden[hidden.size(0)-self.decoder_layers:]
            context = context[context.size(0)-self.decoder_layers:]
        if self.enc_dec_hidden_bridge is not None:
            hidden = self.tanh(self.enc_dec_hidden_bridge(hidden))
            context = self.tanh(self.enc_dec_hidden_bridge(context))
        # context override for debugging the effect of ctx in decoder initialization
        # context = torch.zeros_like(context,  device=device, dtype=torch.float32)
        return hidden, context

    def encoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.encoder_layers * (2 if self.encoder_bidirectional else 1),
                           batch_size, self.encoder_hidden, device=device, dtype=torch.float32)

    def decoder_init(self, batch_size):
        # num_layers * num_directions, batch, hidden_size
        return torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden, device=device, dtype=torch.float32), \
               torch.zeros(self.decoder_layers, batch_size, self.decoder_hidden, device=device, dtype=torch.float32)