"""
This file is a guide on how to start to write a new model which is supposed to work with other parts of the toolkit
"""
import torch
from torch import nn
import torchtext
if torchtext.__version__.startswith('0.9') or torchtext.__version__.startswith('0.10'):
    from torchtext.legacy import data
else:
    from torchtext import data
from configuration import cfg, device
from utils.containers import DecodingSearchNode


class NMTModel(nn.Module):
    def __init__(self, SRC: data.Field, TGT: data.Field):
        """
        :param SRC: the trained torchtext.data.Field object containing the source side vocabulary
        :param TGT: the trained torchtext.data.Field object containing the target side vocabulary
        """
        super(NMTModel, self).__init__()
        self.SRC = SRC
        self.TGT = TGT
        # You may create the loss computation object from any other type. The direct access through the model,
        # guarantees a minimal communication and memory overhead.
        self.criterion = nn.CrossEntropyLoss(ignore_index=TGT.vocab.stoi[cfg.pad_token], reduction='sum')

        self.decoder_layers = int(cfg.decoder_layers)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

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
        input_tensor, input_lengths = input_tensor_with_lengths
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

    def decode(self, input_tensor_with_lengths, output_tensor_with_length=None, test_mode=False, beam_size=1):
        """
        :param input_tensor_with_lengths: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param output_tensor_with_length: tuple(max_seq_length * batch_size, batch_size: actual sequence lengths)
        :param test_mode: a flag indicating whether the model is allowed to use the target tensor for input feeding
        :param beam_size: number of the hypothesis expansions during inference
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

        # #################################CALLING THE ENCODER TO ENCODE THE INPUT BATCH###############################
        decoding_initializer, encoder_output, encoder_memory, attention_mask, target_length = self.encode(
            input_tensor_with_lengths)

        # #################################INITIALIZATION OF DECODING PARAMETERS#######################################
        nodes = [decoding_initializer]
        tokens_count = 0.0
        # #################################ITERATIVE GENERATION OF THE OUTPUT##########################################
        # Iteration over `nodes` and filling out the `result` variable
        return decoding_initializer.result, decoding_initializer.max_attention_indices, \
               decoding_initializer.cumulative_loss,  decoding_initializer.loss_size, tokens_count

    def encoder_init(self, batch_size):
        raise NotImplementedError

    def decoder_init(self, batch_size):
        raise NotImplementedError
