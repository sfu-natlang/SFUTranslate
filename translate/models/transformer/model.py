"""
This file is a guide on how to start to write a new model which is supposed to work with other parts of the toolkit
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
        # You may create the loss computation object from any other type. The direct access through the model,
        # guarantees a minimal communication and memory overhead.
        # self.criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[cfg.pad_token], reduction='sum')

        # #################################### Parameter Initialization ################################################
        # self.decoder_layers = int(cfg.decoder_layers)
        d_model = 512
        h = 8
        dropout = 0.1
        d_ff = 2048
        max_len = 5000
        N = 6
        loss_smoothing = 0.1
        d_model = 256
        h = 8
        dropout = 0.1
        d_ff = 512
        max_len = 5000
        N = 2
        loss_smoothing = 0.1
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

        # ##############################################################################################################

        self.beam_search_decoding = False
        self.beam_size = int(cfg.beam_size)
        self.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
        self.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)

    def init_model_params(self):
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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
        """

        input_tensor, input_lengths = input_tensor_with_lengths
        input_tensor = input_tensor.transpose(0, 1)
        """
        input_sequence_length, batch_size = input_tensor.size()
        target_length = min(int(cfg.maximum_decoding_length * 1.1), input_sequence_length * 2)
        next_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.bos_token]).to(device)
        node_id = 0
        decoder_context = self.decoder_init(batch_size)
        encoder_memory = None
        encoder_output = None
        attention_context = None
        attention_mask = None
        eos_predicted = None
        coverage_vectors = None
        cumulative_predicted_target = None
        max_attention_indices = None
        predicted_target_lm_score = None
        cumulative_loss = torch.zeros(1, device=device)
        loss_size = 0.0
        decoding_initializer = DecodingSearchNode(node_id, decoder_context, next_token, attention_context,
                                                  eos_predicted, coverage_vectors, max_attention_indices,
                                                  cumulative_loss, loss_size, cumulative_predicted_target,
                                                  predicted_target_lm_score)
        return decoding_initializer, encoder_output, encoder_memory, attention_mask, target_length
        """
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
        """input_sequence_length, batch_size = input_tensor_with_lengths[0].size()
        pad_token = torch.LongTensor().new_full((batch_size,), self.TGT.vocab.stoi[cfg.pad_token]).to(device)
        

        # #################################CALLING THE ENCODER TO ENCODE THE INPUT BATCH###############################
        decoding_initializer, encoder_output, encoder_memory, attention_mask, target_length = self.encode(
            input_tensor_with_lengths)

        # #################################INITIALIZATION OF DECODING PARAMETERS#######################################
        nodes = [decoding_initializer]
        tokens_count = 0.0
        # #################################ITERATIVE GENERATION OF THE OUTPUT##########################################
        # Iteration over `nodes` and filling out the `result` variable
        return decoding_initializer.result, decoding_initializer.max_attention_indices, \
               decoding_initializer.cumulative_loss,  decoding_initializer.loss_size, tokens_count"""
        memory, input_mask = self.encode(input_tensor_with_lengths)
        if output_tensor_with_length is not None:
            output_mask = self.generate_tgt_mask(output_tensor)
            x = self.tgt_embed(output_tensor[:, :-1])
            for layer in self.dec_layers:
                x = layer(x, memory, input_mask, output_mask)
            out = self.dec_norm(x)
            # ########################### Code equivalent to SimpleLossCompute #########################################
            y = output_tensor[:, 1:]
            norm = (y != self.TGT.vocab.stoi[cfg.pad_token]).data.sum()
            x = self.generator(out)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1))
            return x, loss, x.size(1), norm
        else:
            self.greedy_decode(input_tensor_with_lengths)

    def greedy_decode(self, input_tensor_with_lengths):
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
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(batch_size, 1).type_as(input_tensor.data).fill_(next_word)], dim=1)
        return ys.transpose(0, 1), torch.zeros(1, device=device), 1, 1

    def generate_src_mask(self, input_tensor):
        return (input_tensor != self.SRC.vocab.stoi[cfg.pad_token]).unsqueeze(-2)

    def generate_tgt_mask(self, output_tensor):
        """Create a mask to hide padding and future words."""
        output_tensor = output_tensor[:, :-1]
        tgt_mask = (output_tensor != self.TGT.vocab.stoi[cfg.pad_token]).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(output_tensor.size(-1)).type_as(tgt_mask.data).clone().detach()
        return tgt_mask
