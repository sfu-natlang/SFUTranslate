"""
This file is the implementation of the transformer encoder decoder model using nn.transformer module in pytorch.
"""
import torch
from torch import nn
from configuration import cfg, device
from models.transformer.optim import LabelSmoothing
from models.transformer.utils import copy
from models.transformer.modules import Embeddings, Generator
import math
from readers.data.field import Field


class PositionalEncoding(nn.Module):
    """
    Implement the PE function (Page 6) in the paper (https://arxiv.org/pdf/1706.03762.pdf)
    It is different from the implemented one in modules.py (the file besides this one) in the dimensions of input it accepts
    The copy has been done to have the legacy code working and preventing lots of .transpose() call for this script to work.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # here is different from the other implementation
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :].clone().detach().requires_grad_(False)
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, SRC: Field, TGT: Field):
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
        # #################################### MODEL INITIALIZATION ####################################################
        default_activation = 'gelu'
        self.backbone = nn.Transformer(d_model, h, N, N, d_ff, dropout, default_activation, None, None)
        # #################################### EMBEDDINGS INITIALIZATION ###############################################
        c = copy.deepcopy
        position = PositionalEncoding(d_model, dropout, max_len)
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
        if output_tensor_with_length is not None and not test_mode:
            return self.decode(input_tensor_with_lengths, output_tensor_with_length)
        elif self.beam_search_decoding:
            return self.beam_search_decode(input_tensor_with_lengths, beam_size=self.beam_size)
        else:
            return self.greedy_decode(input_tensor_with_lengths)

    def decode(self, input_tensor_with_lengths, output_tensor_with_length=None, **kwargs):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :param beam_size: number of the hypothesis expansions during inference
        """
        input_tensor, _ = input_tensor_with_lengths
        batch_size = input_tensor.size(1)
        output_tensor, _ = output_tensor_with_length
        output_tensor_i, output_tensor_o = output_tensor[:-1, :], output_tensor[1:, :]

        src_mask = None  # self.backbone.generate_square_subsequent_mask(input_tensor.size(0)).to(device)
        memory_mask = None
        src_key_padding_mask = (input_tensor == self.SRC.vocab.stoi[cfg.pad_token]).transpose(0, 1).to(device)
        tgt_key_padding_mask = (output_tensor_i == self.TGT.vocab.stoi[cfg.pad_token]).transpose(0, 1).to(device)
        tgt_mask = self.backbone.generate_square_subsequent_mask(output_tensor_i.size(0)).to(device)
        memory_key_padding_mask = src_key_padding_mask.clone()

        ys = torch.ones(1, batch_size).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor.data)
        out = self.backbone(self.src_embed(input_tensor), self.tgt_embed(output_tensor_i),
                            src_mask, tgt_mask, memory_mask, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
        out = self.generator(out)
        loss = self.criterion(out.view(-1, out.size(-1)), output_tensor_o.view(-1))

        max_attention_indices = None
        norm = (output_tensor_o != self.TGT.vocab.stoi[cfg.pad_token]).data.sum()
        for i in range(out.size(0)-1):
            _, next_word = torch.max(out.select(0, i), dim=1)
            ys = torch.cat([ys, next_word.view(1, batch_size)], dim=1)
        return ys, max_attention_indices, loss, out.size(0), float(norm.item())

    def extract_output_probabilities(self, ys, memory, src_key_padding_mask):
        memory_mask = None  # src_mask.clone()
        memory_key_padding_mask = src_key_padding_mask.clone()
        tgt_key_padding_mask = (ys == self.TGT.vocab.stoi[cfg.pad_token]).transpose(0, 1).to(device)
        tgt_mask = self.backbone.generate_square_subsequent_mask(ys.size(0)).to(device)

        output = self.backbone.decoder(self.tgt_embed(ys), memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                       tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        prob = self.generator(output[-1, :])
        return prob

    def greedy_decode(self, input_tensor_with_lengths, **kwargs):
        input_tensor, _ = input_tensor_with_lengths
        input_sequence_length, batch_size = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)

        src_mask = None  # self.backbone.generate_square_subsequent_mask(input_tensor.size(0)).to(device)
        src_key_padding_mask = (input_tensor == self.SRC.vocab.stoi[cfg.pad_token]).transpose(0, 1).to(device)

        ys = torch.ones(1, batch_size).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor.data)
        memory = self.backbone.encoder(self.src_embed(input_tensor), mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        for i in range(target_length-1):
            prob = self.extract_output_probabilities(ys, memory, src_key_padding_mask)
            _, next_word = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word.view(1, batch_size)], dim=0)
        max_attention_indices = None
        return ys, max_attention_indices, torch.zeros(1, device=device), 1, 1

    def beam_search_decode(self, input_tensor_with_lengths, beam_size=1, **kwargs):
        input_tensor, _ = input_tensor_with_lengths
        input_sequence_length, batch_size = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)

        src_mask = None  # self.backbone.generate_square_subsequent_mask(input_tensor.size(0)).to(device)
        src_key_padding_mask = (input_tensor == self.SRC.vocab.stoi[cfg.pad_token]).transpose(0, 1).to(device)

        init_ys = torch.ones(1, batch_size).fill_(self.TGT.vocab.stoi[cfg.bos_token]).type_as(input_tensor.data)
        memory = self.backbone.encoder(self.src_embed(input_tensor), mask=src_mask, src_key_padding_mask=src_key_padding_mask)

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
                prob = self.extract_output_probabilities(ys, memory, src_key_padding_mask)
                k_values, k_indices = torch.topk(prob, dim=1, k=k)
                for beam_index in range(k):
                    overall_index = n_id * k + beam_index
                    all_predictions[:, overall_index] = k_indices[:, beam_index]
                    all_lm_scores[:, overall_index] = lm_scores + k_values[:, beam_index]
            k_values, k_indices = torch.topk(all_lm_scores, dim=1, k=k)
            temp_next_nodes = []
            # creating the next k hypotheses
            for beam_index in range(k):
                node_ids = k_indices[:, beam_index] // k
                node_ids = list(node_ids.cpu().numpy())  # list of size batch_size
                pred_ids = list(k_indices[:, beam_index].cpu().numpy())
                lm_score = k_values[:, beam_index]

                next_word = torch.zeros((batch_size,), device=device).long()
                for b in range(batch_size):
                    next_word[b] = all_predictions[b, pred_ids[b]]

                eos_p = torch.cat(
                    [nodes[n_id][2][b_id].unsqueeze(0) for b_id, n_id in enumerate(node_ids)], dim=0)
                eos_predicted = torch.max(eos_p, (next_word == self.TGT.vocab.stoi[cfg.eos_token]))
                ys = torch.cat([nodes[n_id][0][:, b_id].unsqueeze(1) for b_id, n_id in enumerate(node_ids)], dim=1)
                ys = torch.cat([ys, next_word.view(1, batch_size)], dim=0)
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
                tokens = node[0][:, b_ind]
                eos_ind = torch.nonzero(torch.eq(tokens, self.TGT.vocab.stoi[cfg.eos_token]), as_tuple=False).view(-1)
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