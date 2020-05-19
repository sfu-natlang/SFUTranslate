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
from readers.data_provider import tgt_tokenizer_obj
from utils.evaluation import convert_target_batch_back
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from transformers import BertForMaskedLM


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
        self.multi_head_d_k = d_model // h
        c = copy.deepcopy
        # ling_emb_feature_count should be equal to len(self.head_converter) -1
        enc_attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        encoder_layer = EncoderLayer(d_model, c(enc_attn), c(ff), dropout)
        self.enc_layers = clones(encoder_layer, N-3)
        self.enc_norm = LayerNorm(encoder_layer.size)

        # #################################### DECODER INITIALIZATION ##################################################
        attn = MultiHeadedAttention(h, d_model)
        position = PositionalEncoding(d_model, dropout, max_len)
        decoder_layer = DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout)
        self.dec_layers = clones(decoder_layer, N)
        self.dec_norm = LayerNorm(decoder_layer.size)

        # #################################### EMBEDDINGS INITIALIZATION ###############################################
        self.src_embed = nn.Sequential(Embeddings(d_model, len(SRC.vocab)), c(position))
        self.tgt_embed = nn.Sequential(Embeddings(d_model, len(TGT.vocab)), c(position))
        # #################################### GENERATOR INITIALIZATION ################################################
        self.generator = Generator(d_model, len(TGT.vocab))

        # #################################### BEAM SEARCH PARAMETERS ##################################################
        self.beam_search_decoding = False
        self.beam_size = int(cfg.beam_size)
        self.beam_search_length_norm_factor = float(cfg.beam_search_length_norm_factor)
        self.beam_search_coverage_penalty_factor = float(cfg.beam_search_coverage_penalty_factor)

        # #################################### BERT RELATED PARAMETERS #################################################
        self.augment_input_with_ling_heads = bool(cfg.augment_input_with_ling_heads)
        if self.augment_input_with_ling_heads:
            print("Augmenting Transformer model with linguistic head inputs")
        self.embed_src_with_ling_emb = False if self.augment_input_with_ling_heads else bool(cfg.embed_src_with_ling_emb)
        if self.embed_src_with_ling_emb:
            print("Augmenting Transformer model with linguistic embeddings module")
        self.embed_src_with_bert = False if self.embed_src_with_ling_emb else bool(cfg.embed_src_with_bert)
        if self.embed_src_with_bert:
            print("Augmenting Transformer model with bert embeddings module")
        self.bert_tokenizer = tgt_tokenizer_obj
        self.bert_lm = None
        self.head_converter = None
        self.head_converted_to_d_model = None
        self.d_model = d_model
        self.bert_position = c(position)
        if d_model != 768:
            self.bert_bridge = nn.Linear(768, d_model, bias=False)
        else:
            self.bert_bridge = None
        self.input_gate = nn.Linear(d_model * 2, d_model, bias=True)
        self.number_of_bert_layers = 13
        self.bert_weights_for_average_pooling = nn.Parameter(torch.zeros(self.number_of_bert_layers),
                                                             requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def bert_embed(self, input_tensor):
        """
        :param input_tensor: batch_size * max_seq_length
        """
        # model_name = 'bert-base-uncased'
        # input_sentences = ["This is hassan", "This is hamid"]
        input_sentences = convert_target_batch_back(input_tensor.transpose(0, 1), self.SRC)
        # input_sentences = ["das ist ein arzt"]
        sequences = [torch.tensor(self.bert_tokenizer.encode(input_sentence, add_special_tokens=True), device=device)
                     for input_sentence in input_sentences]
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)
        input_mask = self.generate_src_mask(input_ids)
        outputs = self.bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # 13 * (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.detach().unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                self.softmax(self.bert_weights_for_average_pooling))
        if self.bert_bridge is not None:
            embedded = self.bert_bridge(embedded)

        return self.bert_position(embedded), input_mask

    def get_ling_embed_attention_keys(self, input_tensor):
        """
        :param input_tensor: batch_size * max_seq_length
        """
        # ####################################LOADING THE TRAINED SUB-LAYERS############################################
        input_sentences = convert_target_batch_back(input_tensor.transpose(0, 1), self.SRC)
        sequences = list(map(lambda x: torch.tensor(self.bert_tokenizer.convert_tokens_to_ids(
            x.replace(self.SRC.unk_token, self.bert_tokenizer.unk_token).split()), device=device), input_sentences))
        # ####################################CONVERTING INPUT SEQUENCE TO EMBEDDED BERT################################
        input_ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=self.bert_tokenizer.pad_token_id)
        outputs = self.bert_lm(input_ids, masked_lm_labels=input_ids)[2]  # (batch_size * [input_length + 2] * 768)
        all_layers_embedded = torch.cat([o.unsqueeze(0) for o in outputs], dim=0)
        embedded = torch.matmul(all_layers_embedded.permute(1, 2, 3, 0),
                                self.softmax(self.bert_weights_for_average_pooling)) # [:, 1:-1, :]
        # ##############################################################################################################
        # len(features_list) * batch_size * max_sequence_length, (H/ (len(features_list) + 1))
        keys = [hc(embedded).detach() for hc in self.head_converter[:-1]]  # the last layer contains what we have not considered
        return keys

    def init_model_params(self):
        """
        Initialize parameters with Glorot / fan_avg
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.embed_src_with_bert or self.embed_src_with_ling_emb or self.augment_input_with_ling_heads:
            assert cfg.src_tokenizer == "bert", \
                "data provider should enforce bert tokenizer if bert language model is going to be used here"
            print("Running the init params for bert language model")
            self.bert_lm = BertForMaskedLM.from_pretrained(tgt_tokenizer_obj.model_name, output_hidden_states=True).to(device)
        if self.embed_src_with_ling_emb or self.augment_input_with_ling_heads:
            print("Running the init params for ling_emb [src=german]")
            self.src_embed = nn.Sequential(Embeddings(self.d_model, self.bert_tokenizer.vocab_size), self.src_embed[1]).to(device)
            # ling_emb_data_address = "iwslt_head_conv"
            ling_emb_data_address = cfg.ling_emb_data_address
            so = torch.load(ling_emb_data_address, map_location=lambda storage, loc: storage)
            # features_list = so['features_list']
            self.head_converter = so['head_converters'].to(device)
            # for param in self.head_converter.parameters():
            #    param.requires_grad = False
            # self.head_converter.requires_grad = False
            self.bert_weights_for_average_pooling = nn.Parameter(so['bert_weights'].to(device), requires_grad=True)
            ling_emb_key_size = self.head_converter[0].out_features
            ling_emb_feature_count = len(self.head_converter) - 1
            if ling_emb_feature_count > 0:
                self.head_converted_to_d_model = nn.Linear(ling_emb_key_size * ling_emb_feature_count,
                                                           self.d_model, bias=True).to(device)
            if self.embed_src_with_ling_emb and ling_emb_feature_count > 0:
                ling_emb_bridges = clones(nn.Linear(ling_emb_key_size, self.multi_head_d_k), ling_emb_feature_count)
                c = copy.deepcopy
                for layer in self.enc_layers:
                    layer.self_attn.ling_emb_bridges = c(ling_emb_bridges).to(device)

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
        if self.embed_src_with_bert:
            x, input_mask = self.bert_embed(input_tensor)
        else:
            input_mask = self.generate_src_mask(input_tensor)
            x = self.src_embed(input_tensor)
        if self.embed_src_with_ling_emb:
            # TODO pass these keys to the self attention module
            self_attention_key_list = self.get_ling_embed_attention_keys(input_tensor)
        else:
            self_attention_key_list = []
        if self.augment_input_with_ling_heads:
            x_prime = self.head_converted_to_d_model(torch.cat(self.get_ling_embed_attention_keys(input_tensor), dim=-1))
            x = self.input_gate(torch.cat((x_prime, x), dim=-1))
        for layer in self.enc_layers:
            x = layer(x, input_mask, self_attention_key_list)
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

