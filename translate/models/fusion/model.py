import sys
import numpy as np
import torch
import torchtext
if torchtext.__version__.startswith('0.9'):
    from torchtext.legacy import data
else:
    from torchtext import data
from torch import nn

from configuration import cfg, device
from models.transformer.model import Transformer
from models.transformer.utils import subsequent_mask

import IPython
DEBUG_JETIC = False


class DictionaryFusionTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(DictionaryFusionTransformer, self).__init__(SRC, TGT)
        d_model = int(cfg.transformer_d_model)
        self.PG_L1 = nn.Linear(d_model, 1)
        self.PG_sigmoid = nn.Sigmoid()

        self.PG_V2 = nn.Linear(d_model, 1)
        self.PG_W = nn.Linear(d_model, d_model)
        self.PG_U = nn.Linear(d_model, d_model)

    def decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, beam_size=1, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :param beam_size: number of the hypothesis expansions during inference
        """
        d_model = int(cfg.transformer_d_model)
        # #################################INITIALIZATION OF ENCODING PARAMETERS#######################################
        if output_tensor_with_length is not None:
            output_tensor, outputs_lengths = output_tensor_with_length
            output_tensor = output_tensor.transpose(0, 1)
        else:
            output_tensor, outputs_lengths = None, None

        # #################################CALLING THE ENCODER TO ENCODE THE INPUT BATCH###############################
        memory, input_mask, input_tensor = self.encode(input_tensor_with_lengths, **kwargs)
        # memory: last layer of input sequence: batch_size, input_sequence_length, d
        batch_size, input_sequence_length = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        ys = torch.ones(batch_size, 1).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor.data)

        if output_tensor_with_length is not None and not test_mode:
            batch_size = input_tensor_with_lengths[0].size(1)
            # batch_size
            ys = torch.ones(batch_size, 1).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor_with_lengths[0].data)
            # -- Argmax of greedy decoding
            # -- dim: batch_size, outputs_lengths(decoder)

            output_mask = self.generate_tgt_mask(output_tensor)
            # -- dim: same as output_tensor: max_seq_length, batch_size

            x = self.tgt_embed(output_tensor[:, :-1])
            # -- here it's size of tgt embedding: batch_size, output_tensor_size-1, d
            # #################################  GENERATION OF THE OUTPUT ALL AT ONCE ##################################
            for layer in self.dec_layers:
                x = layer(x, memory, input_mask, output_mask)
            # -- here it's size of tgt embedding: batch_size, output_tensor_size-1, d

            out = self.dec_norm(x)  # take as decoder representation
            # same dimension as x, use as decoder representation

            # ########################### CODE EQUIVALENT TO SIMPLELOSSCOMPUTE #########################################
            y = output_tensor[:, 1:]
            # same as batch_size, output_tensor_size-1
            # reference index
            norm = (y != self.TGT.vocab.stoi[cfg.pad_token]).data.sum()
            # number of non-padded tokens in the reference
            x = self.generator.forward_no_log(out)  # batch_size, output_tensor_size-1, tgt_vocab_size

            # ########################### Jetic's PG stuff #############################################################
            # ########################### Based on LexN1M6 #############################################################
            # Encoded tensors and Decoder tensors
            # encoded: memory, dimension: batch_size, output_tensor_size-1
            # decoded: out
            in_seq_len = memory.size()[1]
            ou_seq_len = out.size()[1]

            p_gen = self.PG_sigmoid(self.PG_L1(out))
            if DEBUG_JETIC:
                print("p_gen[0]", p_gen[0])
            # dimension: batch_size, ou_seq_len, 1

            # score[i][j] = V * tanh(W * h_enc[i] + U * h_dec[i])
            # beta = softmax(score)
            h_enc = self.PG_W(memory).view(batch_size, in_seq_len, 1, d_model)
            h_dec = self.PG_U(out).view(batch_size, 1, ou_seq_len, d_model)
            h_enc = h_enc.repeat(1, 1, ou_seq_len, 1)
            h_dec = h_dec.repeat(1, in_seq_len, 1, 1)
            score = self.PG_V2(torch.tanh(h_enc + h_dec)).view(batch_size, in_seq_len, ou_seq_len)
            beta = nn.functional.softmax(score, dim=1)
            if DEBUG_JETIC:
                print("beta[0, :, 2]", beta[0, :, 2])

            # Loss computation
            local_lex = [np.array(item) for item in kwargs['bilingual_dict']]
            local_lex = [np.pad(item, ((0, in_seq_len - item.shape[0]), (0, ou_seq_len - item.shape[1])), 'constant', constant_values=(0, 0)) for item in local_lex]

            local_lex = torch.tensor([item.tolist() for item in local_lex]).to(device)
            if DEBUG_JETIC:
                print("local_lex[0, :, 2]", local_lex[0, :, 2])
            local_lex = torch.sum((local_lex * beta), dim=1).view(batch_size, ou_seq_len, 1)
            if DEBUG_JETIC:
                print("local_lex[0, 2]", local_lex[0, 2])
            if DEBUG_JETIC:
                print("y[0][2]", y[0][2])
            if DEBUG_JETIC:
                print("x[0][2]", x[0][2])
            lex_x = torch.log(p_gen * x + (1 - p_gen) * local_lex * torch.nn.functional.one_hot(y, 18291)).to(device)

            loss_lex = self.criterion(lex_x.contiguous().view(-1, lex_x.size(-1)), y.contiguous().view(-1))
            loss_dec = self.criterion(torch.log(x).contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            loss = loss_lex + loss_dec

            # ########################### End Jetic's stuff ############################################################

            for i in range(x.size(1) - 1):
                _, next_word = torch.max(x.select(1, i), dim=1)
                ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
            try:
                max_attention_indices = self.compute_maximum_attention_indices(target_length, batch_size)
            except RuntimeError as e:
                print(e, file=sys.stderr)
                max_attention_indices = None
            return ys.transpose(0, 1), max_attention_indices, loss, x.size(1), float(norm.item())
        elif self.beam_search_decoding:
            return self.beam_search_decode(memory, input_mask, input_tensor, ys, batch_size, target_length, beam_size, **kwargs)
        else:
            return self.greedy_decode(memory, input_mask, input_tensor, ys, batch_size, target_length, **kwargs)

    def extract_output_probabilities(self, ys, memory, src_mask, input_tensor, **kwargs):
        output_tensor, output_mask = ys.clone().detach(), subsequent_mask(ys.size(1)).type_as(input_tensor.data).clone().detach()
        batch_size, input_sequence_length = input_tensor.size()
        d_model = int(cfg.transformer_d_model)
        x = self.tgt_embed(output_tensor)
        for layer in self.dec_layers:
            x = layer(x, memory, src_mask, output_mask)
        out = self.dec_norm(x)
        out = out[:, -1]
        prob = self.generator(out)
        p_gen = self.PG_sigmoid(self.PG_L1(out))
        # ######################################## calculating the copy probabilities
        in_seq_len = memory.size()[1]

        h_enc = self.PG_W(memory).view(batch_size, in_seq_len, 1, d_model)
        h_dec = self.PG_U(out).view(batch_size, 1, 1, d_model)
        # h_enc = h_enc.repeat(1, 1, 1, 1)
        h_dec = h_dec.repeat(1, in_seq_len, 1, 1)
        score = self.PG_V2(torch.tanh(h_enc + h_dec)).view(batch_size, in_seq_len, 1)
        beta = nn.functional.softmax(score, dim=1)
        assert 'bilingual_dict' in kwargs
        max_cols = max([len(row) for batch in kwargs['bilingual_dict'] for row in batch])
        max_rows = max([len(batch) for batch in kwargs['bilingual_dict']])

        am_beta = torch.argmax(beta, dim=1)  # batch * max_cols

        local_lex = [[row + [0] * (max_cols - len(row)) for row in batch] for batch in kwargs['bilingual_dict']]  # batch is stc_len * tgt_len
        local_lex = torch.tensor([batch + [[0] * (max_cols)] * (max_rows - len(batch)) for batch in local_lex]).to(device)

        candidate = torch.argmax(torch.stack(
            [local_lex[i][am_beta[i]].unsqueeze(0) for i in range(batch_size)], dim=0).to(device), dim=-1)

        one_h = torch.nn.functional.one_hot(candidate.view(-1), len(self.TGT.vocab)).float().to(device)

        return torch.where(p_gen.repeat(1, len(self.TGT.vocab)) > 0.5, prob, one_h)
