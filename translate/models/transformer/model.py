"""
This file is the implementation of the tranformer encoder decoder model based on The Annotated Transformer
(https://www.aclweb.org/anthology/W18-2509/)
"""
import torch
from torch import nn
from torchtext import data
from configuration import cfg, device
from models.transformer.optim import LabelSmoothing
from utils.containers import DecodingSearchNode
from models.transformer.utils import clones, copy, subsequent_mask
from models.transformer.modules import EncoderLayer, MultiHeadedAttention, PositionwiseFeedForward, PositionalEncoding, \
    LayerNorm, DecoderLayer, Embeddings, Generator


class Transformer(nn.Module):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        """
        :param SRC: the trained torchtext.data.Field object containing the source side vocabulary
        :param TGT: the trained torchtext.data.Field object containing the target side vocabulary
        """
        super(Transformer, self).__init__()
        self.SRC = SRC
        self.TGT = TGT

        # #################################### Parameter Initialization ################################################
        d_model = int(cfg.transformer_d_model)
        h = int(cfg.transformer_h)
        dropout = float(cfg.transformer_dropout)
        d_ff = int(cfg.transformer_d_ff)
        max_len = int(cfg.transformer_max_len)
        N = int(cfg.transformer_N)
        loss_smoothing = float(cfg.transformer_loss_smoothing)

        # #################################### Loss Function Initialization ############################################
        self.criterion = LabelSmoothing(size=len(TGT.vocab),
                                        padding_idx=TGT.vocab.stoi[cfg.pad_token], smoothing=loss_smoothing)

        # #################################### ENCODER INITIALIZATION ##################################################
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)
        self.enc_layers = clones(encoder_layer, N)
        self.enc_norm = LayerNorm(encoder_layer.size)

        # #################################### DECODER INITIALIZATION ##################################################
        position = PositionalEncoding(d_model, dropout, max_len)
        decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
        self.dec_layers = clones(decoder_layer, N)
        self.dec_norm = LayerNorm(decoder_layer.size)

        # #################################### EMBEDDINGS INITIALIZATION ###############################################
        self.src_embed = nn.Sequential(Embeddings(d_model, len(SRC.vocab)), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, len(TGT.vocab)), c(position))
        # #################################### GENERATOR INITIALIZATION ################################################
        self.generator = Generator(d_model, len(TGT.vocab))
        if cfg.share_all_embeddings:
            assert cfg.share_vocabulary, "For sharing embeddings, you need to set the share_vocabulary flag as well!"
            print("Sharing all embedding layers in source and target side ...")
            self.src_embed[0].lut.weight = self.tgt_embed[0].lut.weight
            self.generator.proj.weight = self.tgt_embed[0].lut.weight

        # #################################### BEAM SEARCH PARAMETERS ##################################################
        self.beam_search_decoding = False
        self.beam_size = int(cfg.beam_size)
        self.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
        self.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)

    def init_model_params(self):
        """
        Initialize parameters with Glorot / fan_avg
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        """
        return self.decode(input_tensor_with_lengths, output_tensor_with_length, test_mode, beam_size=self.beam_size)

    def encode(self, input_tensor_with_lengths):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, input_lengths = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        input_mask = self.generate_src_mask(input_tensor)
        x = self.src_embed(input_tensor)
        for layer in self.enc_layers:
            x = layer(x, input_mask)
        return self.enc_norm(x), input_mask

    def decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, beam_size=1):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :param beam_size: number of the hypothesis expansions during inference
        """
        # #################################INITIALIZATION OF ENCODING PARAMETERS#######################################
        if output_tensor_with_length is not None:
            output_tensor, outputs_lengths = output_tensor_with_length
            output_tensor = output_tensor.transpose(0, 1)
        else:
            output_tensor, outputs_lengths = None, None

        # #################################CALLING THE ENCODER TO ENCODE THE INPUT BATCH###############################
        memory, input_mask = self.encode(input_tensor_with_lengths)

        if output_tensor_with_length is not None and not test_mode:
            batch_size = input_tensor_with_lengths[0].size(1)
            ys = torch.ones(batch_size, 1).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor_with_lengths[0].data)
            output_mask = self.generate_tgt_mask(output_tensor)
            x = self.tgt_embed(output_tensor[:, :-1])
            # #################################  GENERATION OF THE OUTPUT ALL AT ONCE ##################################
            for layer in self.dec_layers:
                x = layer(x, memory, input_mask, output_mask)
            out = self.dec_norm(x)
            # ########################### CODE EQUIVALENT TO SIMPLELOSSCOMPUTE #########################################
            y = output_tensor[:, 1:]
            norm = (y != self.TGT.vocab.stoi[cfg.pad_token]).data.sum()
            x = self.generator(out)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            max_attention_indices = None
            for i in range(x.size(1)-1):
                _, next_word = torch.max(x.select(1, i), dim=1)
                ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
            return ys.transpose(0, 1), max_attention_indices, loss, x.size(1), float(norm.item())
        elif self.beam_search_decoding:
            return self.beam_search_decode(input_tensor_with_lengths, beam_size)
        else:
            return self.greedy_decode(input_tensor_with_lengths)

    def greedy_decode(self, input_tensor_with_lengths):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        """
        input_tensor, input_lengths = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        batch_size, input_sequence_length = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        memory, src_mask = self.encode(input_tensor_with_lengths)
        ys = torch.ones(batch_size, 1).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor.data)
        for i in range(target_length-1):
            output_tensor, output_mask = ys.clone().detach(), \
                                              subsequent_mask(ys.size(1)).type_as(input_tensor.data).clone().detach()
            x = self.tgt_embed(output_tensor)
            for layer in self.dec_layers:
                x = layer(x, memory, src_mask, output_mask)
            out = self.dec_norm(x)
            prob = self.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
        max_attention_indices = None
        return ys.transpose(0, 1), max_attention_indices, torch.zeros(1, device=device), 1, 1

    def beam_search_decode(self, input_tensor_with_lengths, beam_size=1):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param beam_size: number of the hypothesis expansions during inference
        """
        input_tensor, input_lengths = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        batch_size, input_sequence_length = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        memory, src_mask = self.encode(input_tensor_with_lengths)

        # #################################INITIALIZATION OF DECODING PARAMETERS#######################################
        init_ys = torch.ones(batch_size, 1).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor.data)
        nodes = [(init_ys, torch.zeros(batch_size, device=device), torch.zeros(batch_size, device=device).bool())]
        final_results = []

        for i in range(target_length-1):
            k = beam_size - len(final_results)
            if k < 1:
                break
            all_predictions = torch.zeros(batch_size, len(nodes) * k, device=device).long()
            all_lm_scores = torch.zeros(batch_size, len(nodes) * k, device=device).float()
            # iterating over all the available hypotheses to expand the beams
            for n_id, (ys, lm_scores, eos_predicted) in enumerate(nodes):
                output_tensor, output_mask = ys.clone().detach(), \
                                             subsequent_mask(ys.size(1)).type_as(input_tensor.data).clone().detach()
                x = self.tgt_embed(output_tensor)
                for layer in self.dec_layers:
                    x = layer(x, memory, src_mask, output_mask)
                out = self.dec_norm(x)
                prob = self.generator(out[:, -1])
                k_values, k_indices = torch.topk(prob, dim=1, k=k)
                for beam_index in range(k):
                    overall_index = n_id * k + beam_index
                    all_predictions[:, overall_index] = k_indices[:, beam_index]
                    all_lm_scores[:, overall_index] = lm_scores + k_values[:, beam_index]
            k_values, k_indices = torch.topk(all_lm_scores, dim=1, k=k)
            temp_next_nodes = []
            # creating the next k hypotheses
            for beam_index in range(k):
                node_ids = k_indices[:, beam_index] / k
                node_ids = list(node_ids.cpu().numpy())  # list of size batch_size
                pred_ids = list(k_indices[:, beam_index].cpu().numpy())
                lm_score = k_values[:, beam_index]

                next_word = torch.zeros((batch_size,), device=device).long()
                for b in range(batch_size):
                    next_word[b] = all_predictions[b, pred_ids[b]]

                eos_p = torch.cat(
                    [nodes[n_id][2][b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                eos_predicted = torch.max(eos_p, (next_word == self.TGT.vocab.stoi[cfg.eos_token]))

                ys = torch.cat([nodes[n_id][0][b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                ys = torch.cat([ys, next_word.view(batch_size, 1)], dim=1)
                next_step_node = (ys, lm_score, eos_predicted)
                if sum(eos_predicted.int()) == batch_size:
                    final_results.append(next_step_node)
                else:
                    temp_next_nodes.append(next_step_node)
            del nodes[:]
            nodes = temp_next_nodes
        if not len(final_results):
            for node in nodes:
                final_results.append(node)
        # creating the final result based on the best scoring hypotheses
        result = torch.zeros(target_length, batch_size, device=device)
        lp = lambda l: ((5 + l) ** self.beam_search_length_norm_factor) / (5 + 1) ** self.beam_search_length_norm_factor
        for b_ind in range(batch_size):
            best_score = float('-inf')
            best_tokens = None
            for node in final_results:
                tokens = node[0][b_ind]
                eos_ind = (tokens == self.TGT.vocab.stoi[cfg.eos_token]).nonzero().view(-1)
                if eos_ind.size(0):
                    tsize = eos_ind[0].item()
                else:
                    tsize = tokens.size(0)
                # based on Google's NMT system paper [https://arxiv.org/pdf/1609.08144.pdf]
                # since coverage is not being tracked here, coverage penalty is not also considered in this formula
                lms = node[1][b_ind].item() / lp(tsize)
                if lms > best_score:
                    best_score = lms
                    best_tokens = tokens
            result[:best_tokens[1:].size(0), b_ind] = best_tokens[1:]
        max_attention_indices = None
        return result, max_attention_indices, torch.zeros(1, device=device), 1, 1

    def generate_src_mask(self, input_tensor):
        """
        :param input_tensor: batch_size * max_seq_length
        """
        return (input_tensor != self.SRC.vocab.stoi[cfg.pad_token]).unsqueeze(-2)

    def generate_tgt_mask(self, output_tensor):
        """
        Create a mask to hide padding and future words
        """
        output_tensor = output_tensor[:, :-1]
        tgt_mask = (output_tensor != self.TGT.vocab.stoi[cfg.pad_token]).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(output_tensor.size(-1)).type_as(tgt_mask.data).clone().detach()
        return tgt_mask
