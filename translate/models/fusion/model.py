import numpy as np
import torch
from torchtext import data
from torch import nn

from configuration import cfg, device
from models.transformer.model import Transformer

import IPython


class DictionaryFusionTransformer(Transformer):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        super(DictionaryFusionTransformer, self).__init__(SRC, TGT)
        d_model = int(cfg.transformer_d_model)
        self.PG_V1 = nn.Linear(d_model, 1)
        self.PG_L1 = nn.Linear(d_model, d_model)

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
            x = self.generator(out)  # batch_size, output_tensor_size-1, tgt_vocab_size

            # ########################### Jetic's PG stuff #############################################################
            # ########################### Based on LexN1M6 #############################################################
            # Encoded tensors and Decoder tensors
            # encoded: memory, dimension: batch_size, output_tensor_size-1
            # decoded: out
            in_seq_len = memory.size()[1]
            ou_seq_len = out.size()[1]

            p_gen = self.PG_V1(torch.tanh(self.PG_L1(out)))
            # dimension: batch_size, ou_seq_len, 1

            # score[i][j] = V * tanh(W * h_enc[i] + U * h_dec[i])
            # beta = softmax(score)
            h_enc = self.PG_W(memory).view(batch_size, in_seq_len, 1, d_model)
            h_dec = self.PG_U(out).view(batch_size, 1, ou_seq_len, d_model)
            h_enc = h_enc.repeat(1, 1, ou_seq_len, 1)
            h_dec = h_dec.repeat(1, in_seq_len, 1, 1)
            score = self.PG_V2(torch.tanh(h_enc + h_dec)).view(batch_size, in_seq_len, ou_seq_len)
            beta = nn.functional.softmax(score, dim=1)

            # Loss computation
            lex_x = p_gen * x  # batch_size, ou_seq_len, tgt_vocab_size
            p_lex = 1 - p_gen  # dimension: batch_size, ou_seq_len, 1
            local_lex = [np.array(item) for item in kwargs['bilingual_dict']]
            local_lex = [np.pad(item, ((0, in_seq_len - item.shape[0]), (0, ou_seq_len - item.shape[1])), 'constant', constant_values=(0, 0)) for item in local_lex]

            local_lex = torch.tensor([item.tolist() for item in local_lex])
            local_lex = torch.sum((local_lex * beta), dim=1).view(batch_size, ou_seq_len, 1)
            p_lex = p_lex * local_lex
            lex_x = lex_x + p_lex

            loss_lex = self.criterion(lex_x.contiguous().view(-1, lex_x.size(-1)), y.contiguous().view(-1))
            loss_dec = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            loss = (loss_lex + loss_dec) / 2

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


'''
    def forward(self,
                input_tensor_with_lengths,
                output_tensor_with_length=None,
                test_mode=False,
                **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :return decoding_initializer_result_tensor (output of softmax),
                decoding_initializer_max_attention_indices (can be None),
                decoding_initializer_cumulative_loss,
                decoding_initializer_loss_size,
                batch_tokens_count
        """
        if not test_mode:
            bilingual_dict = kwargs['bilingual_dict']
            # Size: batch_size, max_seq_length,
            IPython.embed()
            # TODO use the bilingual dict in here
            # You may want to convert 'bilingual_dict' into a torch tensor
        return self.decode(input_tensor_with_lengths,
                           output_tensor_with_length,
                           test_mode,
                           beam_size=self.beam_size,
                           **kwargs)
'''
