"""
This file is an implementation of vanilla attentional sequence to sequence models with coverage mechanism
"""
import math
import torch
from torch import nn
import torchtext
if torchtext.__version__.startswith('0.9') or torchtext.__version__.startswith('0.10'):
    from torchtext.legacy import data
else:
    from torchtext import data
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from configuration import cfg, device
from utils.containers import DecodingSearchNode


class STS(nn.Module):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        """
        :param SRC: the trained torchtext.data.Field object containing the source side vocabulary
        :param TGT: the trained torchtext.data.Field object containing the target side vocabulary
        """
        super(STS, self).__init__()
        self.SRC = SRC
        self.TGT = TGT
        # You may create the loss computation object from any other type. The direct access through the model,
        # guarantees a minimal communication and memory overhead.
        self.criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[cfg.pad_token], reduction='sum')
        assert not cfg.share_all_embeddings, "Embedding sharing is not currently supported in STS model."
        # ##############################################################################################################
        self.bahdanau_attention = bool(cfg.bahdanau_attention)
        print("Creating the Seq2Seq Model with {} attention".format("Bahdanau" if self.bahdanau_attention else "Loung"))
        self.coverage = bool(cfg.coverage_required)
        print("Coverage model (See et al. definition [P17-1099]) is {}".format(
            "also considered" if self.coverage else "not considered"))
        self.encoder_layers = int(cfg.encoder_layers)
        self.encoder_bidirectional = True
        self.encoder_hidden = int(cfg.encoder_hidden_size)
        self.encoder_actual_hidden = self.encoder_hidden * (2 if self.encoder_bidirectional else 1)

        self.decoder_layers = int(cfg.decoder_layers)
        self.decoder_hidden = int(cfg.decoder_hidden_size)
        self.decoder_input_size = int(cfg.decoder_emb_size) + self.encoder_actual_hidden

        # ####################################### ESSENTIAL MODEL MODULES###############################################
        self.encoder_emb = nn.Embedding(len(SRC.vocab), int(cfg.encoder_emb_size),
                                        padding_idx=SRC.vocab.stoi[cfg.pad_token])
        self.decoder_emb = nn.Embedding(len(TGT.vocab), int(cfg.decoder_emb_size),
                                        padding_idx=TGT.vocab.stoi[cfg.pad_token])
        self.encoder = nn.LSTM(int(cfg.encoder_emb_size), self.encoder_hidden, self.encoder_layers,
                               bidirectional=self.encoder_bidirectional,
                               dropout=float(cfg.encoder_dropout_rate) if self.encoder_layers > 1 else 0.0)
        self.decoder = nn.LSTM(self.decoder_input_size, self.decoder_hidden, self.decoder_layers,
                               dropout=float(cfg.decoder_dropout_rate) if self.decoder_layers > 1 else 0.0)
        self.out = nn.Linear(self.encoder_actual_hidden + self.decoder_hidden + int(cfg.decoder_emb_size),
                             len(TGT.vocab))
        if self.encoder_actual_hidden != self.decoder_hidden:
            self.enc_dec_hidden_bridge = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1),
                                                   self.decoder_hidden, bias=False)
        else:
            self.enc_dec_hidden_bridge = None

        # ####################################### ATTENTION RELATED MODULES ############################################
        self.attention_W = nn.Linear(self.encoder_hidden * (2 if self.encoder_bidirectional else 1),
                                     self.decoder_hidden, bias=False)
        if self.bahdanau_attention:
            self.attention_U = nn.Linear(self.decoder_hidden, self.decoder_hidden, bias=True)
            self.attention_V = nn.Linear(self.decoder_hidden, 1, bias=False)
        else:
            self.attention_U = None
            self.attention_V = None

        # ####################################### COVERAGE RELATED MODULES #############################################
        if self.coverage:
            if not self.bahdanau_attention:
                raise ValueError("Coverage model is just integrated with Bahdanau Attention")
            self.coverage_lambda = float(cfg.coverage_lambda)
            self.attention_C = nn.Linear(1, self.decoder_hidden, bias=False)
            self.coverage_dropout = nn.Dropout(p=float(cfg.coverage_dropout))
        else:
            self.coverage_lambda = 0.0
            self.attention_C = None
            self.coverage_dropout = None

        # ##############################################################################################################
        self.emb_dropout = nn.Dropout(p=float(cfg.emb_dropout))
        self.out_dropout = nn.Dropout(p=float(cfg.out_dropout))
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        # ##############################################################################################################
        self.beam_search_decoding = False
        self.beam_size = int(cfg.beam_size)
        self.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
        self.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)

    def forward(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        """
        return self.decode(input_tensor_with_lengths, output_tensor_with_length, test_mode, beam_size=self.beam_size)

    def encode(self, input_tensor_with_lengths):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :return decoding_initializer: a DecodingSearchNode initialized with all of the encoder representation nformation
            required for initializing the decoder
        :return decoder_context: the encoder hidden vectors which can be used by the decoder in attention mechanism if
            any available
        :return encoder_output: a tensor containing all the raw encoder outputs in all the time-steps
        :return encoder_memory: a tensor containing all the encoder outputs in all the time-steps preprocessed for
            decoding purposes
        :return attention_context: the computed attention vector based on the initial value of the :return next_token:\
            which has already been filled with start of the sentence token
        :return attention_mask: the masking vector used to zero out the unintentional scores assigned to the padding
            input sentence tokens
        :return eos_predicted: a boolean vector tracking whether the end of sentence has already been predicted in in
            the sentence. This vector is useful in keeping track of the decoding and halting the decoding loop once all
            the sentences in the batch have generated the eos token.
        :return coverage_vectors: a matrix place holder for collecting the attentions scores over the input sequences in
            the batch
        :return predicted_target: a place holder to collect the target sequence predictions of the current step in the
            decoding process
        :return predicted_target_lm_score: a place holder to collect the language model scores predicted for the current
            step predictions in the decoding process
        :return cumulative_predicted_target: a place holder to collect the target sequence predictions step by step as
            the decoding proceeds
        :return max_attention_indices: a matrix place holder for collecting the source token ids gaining maximum values
            of attention in each decoding step
        :return cumulative_loss: a place holder for the cumulative loss of the decoding as the iterations proceed
        :return loss_size: the counter of decoding steps the loss of which is collected in :return cumulative_loss:
        """
        # ######################################### INITIALIZING ENCODER VARIABLES #####################################
        input_tensor, input_lengths = input_tensor_with_lengths
        input_sequence_length, batch_size = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        next_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.bos_token]).to(device)
        init_node_id = 0

        # ######################################### ENCODING THE INPUT TENSOR ##########################################
        embedded_input = self.emb_dropout(self.encoder_emb(input_tensor))  # seq_length × batch_size × emd_size
        packed_input = pack_padded_sequence(embedded_input, input_lengths, enforce_sorted=False)
        encoded_pack_output, encoder_context = self.encoder(packed_input)
        encoder_output, _ = pad_packed_sequence(encoded_pack_output)

        # ######################################### INITIALIZING DECODER VARIABLES #####################################
        decoder_context = self.reformat_encoder_hidden_states(encoder_context)
        attention_mask = input_tensor.transpose(0, 1).unsqueeze(1) != self.SRC.vocab.stoi[cfg.pad_token]
        if self.bahdanau_attention:
            encoder_memory = self.attention_W(encoder_output).transpose(0, 1)
        else:
            encoder_memory = self.attention_W(encoder_output).permute(1, 2, 0)
        attention_context = torch.zeros(batch_size, self.encoder_actual_hidden, device=device)
        eos_predicted = torch.zeros(batch_size, device=device).bool()
        coverage_vectors = torch.zeros(batch_size, input_sequence_length, 1, device=device).float() \
            if self.coverage else None
        max_attention_indices = torch.zeros(target_length, batch_size, device=device)
        cumulative_predicted_target = torch.zeros((batch_size, 1), device=device).long()
        predicted_target_lm_score = torch.zeros((batch_size,), device=device)
        cumulative_loss = 0.0
        loss_size = 0.0
        decoding_initializer = DecodingSearchNode(init_node_id, decoder_context, next_token, attention_context,
                                                  eos_predicted, coverage_vectors, max_attention_indices,
                                                  cumulative_loss, loss_size, cumulative_predicted_target,
                                                  predicted_target_lm_score)
        return decoding_initializer, encoder_output, encoder_memory, attention_mask, target_length

    def decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, beam_size=1):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :param beam_size: number of the hypothesis expansions during inference
        """
        if self.beam_search_decoding:
            return self.beam_search_decode(input_tensor_with_lengths, beam_size=beam_size)
        else:
            return self.greedy_decode(input_tensor_with_lengths, output_tensor_with_length, test_mode)

    def greedy_decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        """
        # #################################INITIALIZATION OF ENCODING PARAMETERS#######################################
        input_sequence_length, batch_size = input_tensor_with_lengths[0].size()
        pad_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.pad_token]).to(device)
        if output_tensor_with_length is not None:
            output_tensor, outputs_lengths = output_tensor_with_length
            tokens_count = float(outputs_lengths.sum().item())
        else:
            output_tensor, outputs_lengths = None, None
            tokens_count = 0.0
        predicted_tokens_count = 0.0

        # #################################CALLING THE ENCODER TO ENCODE THE INPUT BATCH###############################
        decoding_initializer, encoder_output, encoder_memory, attention_mask, target_length = self.encode(
            input_tensor_with_lengths)
        result = torch.zeros(target_length, batch_size, device=device)

        # #################################INITIALIZATION OF DECODING PARAMETERS#######################################
        next_token = decoding_initializer.next_token
        decoder_context = decoding_initializer.decoder_context
        c_t = decoding_initializer.attention_context
        max_attention_indices = decoding_initializer.max_attention_indices
        coverage_vector = decoding_initializer.coverage_vectors
        eos_predicted = decoding_initializer.eos_predicted
        cumulative_loss = decoding_initializer.cumulative_loss
        loss_size = decoding_initializer.loss_size

        # #################################ITERATIVE GENERATION OF THE OUTPUT##########################################
        for t in range(target_length):
            p_vocab, query, decoder_context = self.next_target_distribution(next_token, decoder_context, c_t, batch_size)
            alphas = self.compute_attention_scores(query, encoder_memory,
                                                   attention_mask, input_sequence_length, coverage_vector)
            max_attention_indices[t, :] = alphas.max(dim=-1)[1].view(-1).detach()  # batch_size
            c_t = (alphas @ encoder_output.transpose(0, 1)).squeeze(1)
            greedy_prediction = torch.argmax(p_vocab, dim=1).detach()
            next_token = greedy_prediction  # greedy approach
            eos_predicted = torch.max(eos_predicted, (greedy_prediction == self.TGT.vocab.stoi[cfg.eos_token]))
            if output_tensor is not None:
                if not test_mode:  # Input Feeding
                    next_token = output_tensor.select(0, t + 1) if t < output_tensor.size(0) - 1 else pad_token
                cumulative_loss += self.criterion(p_vocab, next_token)
                loss_size += 1.0
            predicted_tokens_count += batch_size - eos_predicted.sum().item()
            if sum(eos_predicted.int()) == batch_size:
                break
            result[t, :] = greedy_prediction
            if self.coverage:
                cvg_formatted_alphas = alphas.squeeze(1)
                coverage_vector = coverage_vector + cvg_formatted_alphas.unsqueeze(-1)
                if output_tensor is not None:
                    masked_coverage = coverage_vector.squeeze(2) * attention_mask.float().squeeze(1)
                    min_coverage_and_attn = torch.min(masked_coverage, cvg_formatted_alphas)
                    cumulative_loss = cumulative_loss + self.coverage_lambda * min_coverage_and_attn.sum()
        return result, max_attention_indices, cumulative_loss,  loss_size, tokens_count

    def beam_search_decode(self, input_tensor_with_lengths, beam_size=1):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param beam_size: number of the hypothesis expansions during inference
        """
        # #################################INITIALIZATION OF ENCODING PARAMETERS#######################################
        input_sequence_length, batch_size = input_tensor_with_lengths[0].size()
        tokens_count = 0.0

        # #################################CALLING THE ENCODER TO ENCODE THE INPUT BATCH###############################
        decoding_initializer, encoder_output, encoder_memory, attention_mask, target_length = self.encode(
            input_tensor_with_lengths)

        # #################################INITIALIZATION OF DECODING PARAMETERS#######################################
        nodes = [decoding_initializer]
        last_created_node_id = decoding_initializer.id
        final_results = []

        # #################################ITERATIVE GENERATION OF THE OUTPUT##########################################
        for step in range(target_length):
            k = beam_size - len(final_results)
            if k < 1:
                break
            all_predictions = torch.zeros(batch_size, len(nodes) * k, device=device).long()
            all_lm_scores = torch.zeros(batch_size, len(nodes) * k, device=device).float()
            # iterating over all the available hypotheses to expand the beams
            for n_id, node in enumerate(nodes):
                p_vocab, query, decoder_context = self.next_target_distribution(
                    node.next_token, node.decoder_context, node.attention_context, batch_size)
                node.set_result(self.softmax(p_vocab), query, decoder_context)
                k_values, k_indices = torch.topk(node.result_output, dim=1, k=k)
                for beam_index in range(k):
                    overall_index = n_id * k + beam_index
                    all_predictions[:, overall_index] = k_indices[:, beam_index]
                    all_lm_scores[:, overall_index] = node.predicted_target_lm_score + torch.log(k_values[:, beam_index])
            # selecting the top k generated indices to filter out the expansion
            k_values, k_indices = torch.topk(all_lm_scores, dim=1, k=k)
            temp_next_nodes = []
            # creating the next k hypotheses
            for beam_index in range(k):
                node_ids = k_indices[:, beam_index] // k
                node_ids = list(node_ids.cpu().numpy())  # list of size batch_size
                pred_ids = list(k_indices[:, beam_index].cpu().numpy())
                lm_score = k_values[:, beam_index]
                last_created_node_id += 1
                query = torch.cat(
                    [nodes[n_id].result_query[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                if self.coverage:
                    coverage_vectors = torch.cat(
                        [nodes[n_id].coverage_vectors[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                else:
                    coverage_vectors = None
                max_attention_indices = torch.cat([nodes[n_id].max_attention_indices[:, b_id].unsqueeze(1)
                                                   for b_id, n_id in enumerate(node_ids)], dim=1)
                alphas = self.compute_attention_scores(query, encoder_memory, attention_mask,
                                                       input_sequence_length, coverage_vectors)
                if self.coverage:
                    cvg_formatted_alphas = alphas.squeeze(1)
                    coverage_vectors = coverage_vectors + cvg_formatted_alphas.unsqueeze(-1)
                # input_tensor => max_seq_length * batch_size
                max_attention_indices[step, :] = alphas.max(dim=-1)[1].view(-1).detach()  # batch_size
                attention_context = (alphas @ encoder_output.transpose(0, 1)).squeeze(1)
                greedy_prediction = torch.zeros((batch_size,), device=device).long()
                for b in range(batch_size):
                    greedy_prediction[b] = all_predictions[b, pred_ids[b]]
                decoder_context = (torch.cat([nodes[n_id].result_context[0][:, b_id, :]
                                              for b_id, n_id in enumerate(node_ids)], dim=0).unsqueeze(0),
                                   torch.cat([nodes[n_id].result_context[1][:, b_id, :]
                                              for b_id, n_id in enumerate(node_ids)], dim=0).unsqueeze(0))
                eos_p = torch.cat(
                    [nodes[n_id].eos_predicted[b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                eos_predicted = torch.max(eos_p, (greedy_prediction == self.TGT.vocab.stoi[cfg.eos_token]))
                prev_tokens = torch.cat(
                    [nodes[n_id].cumulative_predicted_targets[b_id].unsqueeze(0)
                     for b_id, n_id in enumerate(node_ids)], dim=0)
                cumulative_predicted_targets = torch.cat((prev_tokens, greedy_prediction.unsqueeze(-1)), dim=1)
                c_beam = DecodingSearchNode(last_created_node_id, decoder_context, greedy_prediction, attention_context,
                                            eos_predicted, coverage_vectors, max_attention_indices, 0.0, 1.0,
                                            cumulative_predicted_targets, lm_score)

                if sum(eos_predicted.int()) == batch_size:
                    final_results.append(c_beam)
                else:
                    temp_next_nodes.append(c_beam)
            del nodes[:]
            nodes = temp_next_nodes
        if not len(final_results):
            for node in nodes:
                final_results.append(node)
        # creating the final result based on the best scoring hypotheses
        result = torch.zeros(target_length, batch_size, device=device)
        max_attention_indices = torch.zeros(target_length, batch_size, device=device)
        lp = lambda l: ((5 + l) ** self.beam_search_length_norm_factor) / (5 + 1) ** self.beam_search_length_norm_factor
        for b_ind in range(batch_size):
            best_score = float('-inf')
            best_tokens = None
            best_att = None
            for node in final_results:
                tokens = node.cumulative_predicted_targets[b_ind]
                eos_ind = (tokens == self.TGT.vocab.stoi[cfg.eos_token]).nonzero().view(-1)
                if eos_ind.size(0):
                    tsize = eos_ind[0].item()
                else:
                    tsize = tokens.size(0)
                # based on Google's NMT system paper [https://arxiv.org/pdf/1609.08144.pdf]
                if self.coverage:
                    cp = sum([math.log(min(x, 1.0)) for x in
                          list(node.coverage_vectors[b_ind].view(-1).cpu().numpy()) if x > 0.0])
                else:
                    # might not be the best thing to do
                    cp = 0.0
                lms = node.predicted_target_lm_score[b_ind].item() / lp(tsize) \
                    + self.beam_search_coverage_penalty_factor * cp
                if lms > best_score:
                    best_score = lms
                    best_tokens = tokens
                    best_att = node.max_attention_indices[:, b_ind]
            result[:best_tokens[1:].size(0), b_ind] = best_tokens[1:]
            max_attention_indices[:, b_ind] = best_att
        return result, max_attention_indices, torch.zeros(1, device=device),  1.0, tokens_count

    def compute_attention_scores(self, query, memory, attention_mask, input_sequence_length, coverage_vector=None):
        """
        :param query: (batch_size, self.decoder_hidden)
        :param memory: (batch_size, input_sequence_length, self.encoder_hidden * [2 if bidirectional else 1])
        :param attention_mask: (batch_size, 1, input_sequence_length)
        :param coverage_vector: (batch_size, input_sequence_length, 1)
        :param input_sequence_length: the maximum size of input sequences in the current batch
        """
        if self.bahdanau_attention:
            attention_inputs = self.attention_U(query.unsqueeze(1).repeat(1, input_sequence_length, 1)) + memory
            if coverage_vector is not None:
                attention_inputs = attention_inputs + self.attention_C(coverage_vector)
            alphas = self.attention_V(self.tanh(attention_inputs)).squeeze(2).unsqueeze(1)  # b_size, 1, input_len
        else:  # Loung general
            alphas = query.unsqueeze(1) @ memory  # b_size,1,input_len
        alphas = torch.where(attention_mask, alphas, alphas.new_full([1], float('-inf')))
        return self.softmax(alphas)  # batch_size * 1 * max_input_length

    def next_target_distribution(self, next_token, prev_decoder_context, c_t, batch_size):
        """
        :param next_token: (batch_size, ) a single dimensional vector
        :param prev_decoder_context: tuple of size 2: ((1, batch_size, self.decoder_hidden),
                                                        (1, batch_size, self.decoder_hidden))
        :param c_t: (batch_size, self.encoder_hidden * [2 if bidirectional else 1])
        :param batch_size: the number of sentences passed to be translated
        """
        dec_emb = self.emb_dropout(self.decoder_emb(next_token))  # batch_size * decoder_emb_size
        decoder_input = torch.cat([dec_emb, c_t], dim=1).view(1, batch_size, self.decoder_input_size)
        _, decoder_context = self.decoder(decoder_input, prev_decoder_context)
        query = decoder_context[0][-1].view(batch_size, self.decoder_hidden)
        semi_output = self.out_dropout(torch.cat([dec_emb, query, c_t], dim=1))
        # out: batch_size, target_vocab_size
        return self.out(self.tanh(semi_output)).view(batch_size, len(self.TGT.vocab)),  query, decoder_context

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
