"""
The sequence to sequence module with an additional functionality to compute the loss value in the forward path
 (the loss is added to module for speed optimization purposes). The module can be configured via setting the following
  values in the config file to the desired values.
##################################################
trainer:
    model:
        type: seq2seq
        tfr: 1.001 # teacher forcing ratio
        bienc: true # bidirectional encoding
        hsize: 256 # hidden state size of RNN layers
        nelayers: 1 # number of hidden layers in encoder
        ndlayers: 1 # number of hidden layers in decoder
        bsize: 64 # size of the training sentence batches
        ddropout: 0.1 # the dropout probability in the decoder
        init_val: 0.1 # the value to range of which random variables get initiated
        decoder_weight_tying: false # whether the weights need to be tied between the decoder embedding and generator
##################################################
"""
import random
from translate.learning.modules.rnn.decoder import DecoderRNN
from translate.learning.modules.rnn.encoder import EncoderRNN
from typing import List, Any, Tuple, Dict

from translate.backend.utils import backend, list_to_long_tensor, device, zeros_tensor, row_wise_batch_copy
from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
from translate.learning.modules.mlp.generator import GeneratorNN
from translate.learning.models.rnn.results import GreedyDecodingResult, BeamDecodingResult
from translate.readers.datareader import AbsDatasetReader
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


class SequenceToSequence(AbsCompleteModel):
    def __init__(self, configs: ConfigLoader, train_dataset: AbsDatasetReader):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param train_dataset: the dataset from which the statistics regarding dataset will be looked up during
         model configuration
        """
        super(SequenceToSequence, self).__init__(backend.nn.NLLLoss(
            ignore_index=train_dataset.target_vocabulary.get_pad_word_index()))
        self.dataset = train_dataset
        self.teacher_forcing_ratio = configs.get("trainer.model.tfr", 1.1)
        self.auto_teacher_forcing_ratio = configs.get("trainer.model.auto_tfr", False)
        if self.auto_teacher_forcing_ratio:
            self.teacher_forcing_ratio = 1.1
            logger.info("Teacher forcing is set to auto-decay mode!")
        self.bidirectional_encoding = configs.get("trainer.model.bienc", True)
        hidden_size = configs.get("trainer.model.hsize", must_exist=True)
        n_e_layers = configs.get("trainer.model.nelayers", 1)
        n_d_layers = configs.get("trainer.model.ndlayers", 1)
        decoder_dropout = configs.get("trainer.model.ddropout", 0.1)
        init_val = configs.get("trainer.model.init_val", 0.01)
        self.batch_size = configs.get("trainer.model.bsize", must_exist=True)
        decoder_weight_tying = configs.get("trainer.model.decoder_weight_tying", False)

        self.beam_size = configs.get("trainer.model.beam_size", 1)
        assert self.beam_size >= 1
        if self.beam_size == 1:
            logger.info("The validation would be performed using Greedy Decoding")
        else:
            logger.info("The validation would be performed using Beam Search Decoding with Beam Size %d" %
                        self.beam_size)
        self.max_decoding_length = train_dataset.max_sentence_length()
        self.sos_token_id = train_dataset.target_vocabulary.get_begin_word_index()
        self.eos_token_id = train_dataset.target_vocabulary.get_end_word_index()
        self.pad_token_id = train_dataset.target_vocabulary.get_pad_word_index()
        attention_method = configs.get("trainer.model.decoder_attention_method", 'dot')
        attention_type = configs.get("trainer.model.decoder_attention_type", 'global')
        local_attention_d = configs.get("trainer.model.decoder_local_attention_d", 0.0)
        self.encoder = EncoderRNN(len(train_dataset.source_vocabulary),
                                  hidden_size, self.bidirectional_encoding, n_e_layers, self.batch_size,
                                  train_dataset.source_vocabulary.get_pad_word_index())
        self.decoder = DecoderRNN(hidden_size, len(train_dataset.target_vocabulary), self.bidirectional_encoding,
                                  n_d_layers, self.batch_size, decoder_dropout, attention_method, attention_type,
                                  local_attention_d, self.pad_token_id)
        self.generator = GeneratorNN(self.decoder.get_hidden_size(), len(train_dataset.target_vocabulary),
                                     decoder_dropout, needs_additive_bias=not decoder_weight_tying)
        self.encoder_output_size = self.encoder.hidden_size
        if self.bidirectional_encoding:
            self.encoder_output_size *= 2
        if decoder_weight_tying:
            # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
            # https://arxiv.org/abs/1608.05859
            # and
            # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
            # https://arxiv.org/abs/1611.01462
            logger.info("Tying the weights in Seq2Seq.Decoder")
            self.generator.out.weight = self.decoder.embedding.weight
        logger.info("Randomly initiating model variables in the range [-{0}, {0}]".format(init_val))
        for p_set in self.optimizable_params_list():
            for p in p_set:
                p.data.uniform_(-init_val, init_val)

    def encode(self, input_variable: backend.Tensor):
        """
        :param input_variable: 2-D Tensor of size [batch_size, max_input_length (variable for each batch)]
        """
        return self.encoder(input_variable.transpose(0, 1), self.encoder.init_hidden(batch_size=input_variable.size(0)))

    def forward(self, input_variable: backend.Tensor, target_variable: backend.Tensor, *args, **kwargs) \
            -> Tuple[backend.Tensor, int, List[Any]]:
        """
        :param input_variable: 2-D Tensor of size [batch_size, max_input_length (variable for each batch)]
        :param target_variable: 2-D Tensor of size [batch_size, max_input_length (variable for each batch)]
        """
        encoder_outputs, encoder_hidden_params = self.encode(input_variable)
        return self.greedy_decode(encoder_outputs, encoder_hidden_params, target_variable)

    def greedy_decode(self, encoder_outputs: backend.Tensor, encoder_hidden_params: Tuple[backend.Tensor],
                      target_variable: backend.Tensor=None):
        """
        :param encoder_outputs: 3-D Tensor [max_length (might vary per batch), batch_size, hidden_size]
        :param encoder_hidden_params: Pair of size 2 of 3-D Tensors
          [num_enc_dirs*n_enc_layers, batch_size, hidden_size//n_enc_dirs]
        :param target_variable:  2-D Tensor of size [batch_size, max_input_length (variable for each batch)]
         in case of decoding a test sentence could be None
        :return:
        """
        batch_size = encoder_outputs.size(1)
        if target_variable is not None:
            target_variable = target_variable.transpose(0, 1)
            expected_target_length = target_variable.size(0)
        else:
            expected_target_length = self.max_decoding_length
        decoder_input = list_to_long_tensor([self.sos_token_id] * batch_size).to(device)
        # decoder_hidden_params = self.decoder.init_hidden(batch_size=batch_size)
        decoder_hidden_params = self.decoder.reformat_encoder_hidden_states(encoder_hidden_params)
        # previous attended output is initiated to zeros
        h_hat = zeros_tensor(1, batch_size, self.encoder_output_size).squeeze(0)
        result = GreedyDecodingResult(batch_size, self.pad_token_id, self.eos_token_id)
        loss = zeros_tensor(1, 1, 1).view(-1)
        target_length = 0
        while not result.decoding_completed and target_length < 2 * expected_target_length:
            decoder_output, decoder_hidden_params, decoder_attention = \
                self.decoder(decoder_input, decoder_hidden_params, h_hat, encoder_outputs, batch_size=batch_size)
            h_hat = decoder_output
            decoder_output = self.generator(decoder_output)
            if target_variable is not None:
                di = target_length if target_length < expected_target_length \
                    else expected_target_length - 1
                loss += self.criterion(decoder_output, target_variable[di])
            next_decoder_input = result.append(*decoder_output.data.topk(1))
            if random.random() < self.teacher_forcing_ratio and target_variable is not None:
                decoder_input = target_variable[di]  # Teacher forcing
            else:
                decoder_input = next_decoder_input
            target_length += 1
        if target_variable is not None:
            return loss, target_length, result.ids
        else:
            return result.log_probability, target_length, result.ids

    def beam_search_decode(self, encoder_outputs: backend.Tensor, encoder_hidden_params: Tuple[backend.Tensor],
                           target_variable: backend.Tensor=None, k: int=1):
        """
        :param encoder_outputs: 3-D Tensor [max_length (might vary per batch), batch_size, hidden_size]
        :param encoder_hidden_params: Pair of size 2 of 3-D Tensors
          [num_enc_dirs*n_enc_layers, batch_size, hidden_size//n_enc_dirs]
        :param target_variable:  2-D Tensor of size [batch_size, max_input_length (variable for each batch)]
         in case of decoding a test sentence could be None
        :param k: the beam size used for the beam search
        :return:
        """
        assert k > 0
        batch_size = encoder_outputs.size(1)
        if target_variable is not None:
            target_variable = target_variable.transpose(0, 1)
            expected_target_length = target_variable.size(0)
        else:
            expected_target_length = self.max_decoding_length
        encoder_outputs = row_wise_batch_copy(encoder_outputs, k)
        decoder_input = list_to_long_tensor([self.sos_token_id] * (batch_size * k)).to(device)
        # decoder_hidden_params = self.decoder.init_hidden(batch_size=batch_size)
        decoder_hidden_params = self.decoder.reformat_encoder_hidden_states(encoder_hidden_params)
        decoder_hidden_params = (row_wise_batch_copy(decoder_hidden_params[0], k),
                                 row_wise_batch_copy(decoder_hidden_params[1], k))
        # previous attended output is initiated to zeros
        h_hat = zeros_tensor(1, batch_size * k, self.encoder_output_size).squeeze(0)
        result = BeamDecodingResult(batch_size, k, self.pad_token_id, self.eos_token_id)
        loss = zeros_tensor(1, 1, 1).view(-1)
        target_length = 0
        while not result.decoding_completed and target_length < 2 * expected_target_length:
            decoder_output, decoder_hidden_params, decoder_attention = \
                self.decoder(decoder_input, decoder_hidden_params, h_hat, encoder_outputs, batch_size=batch_size * k)
            generator_output = self.generator(decoder_output)
            if target_variable is not None:
                di = target_length if target_length < expected_target_length \
                    else expected_target_length - 1
                loss += self.criterion(generator_output, row_wise_batch_copy(target_variable[di], k))
            next_decoder_input, selection_bucket = result.append(*generator_output.data.topk(k))
            h_hat = backend.stack([decoder_output[ind] for ind in selection_bucket], dim=0)
            decoder_hidden_params = (backend.stack([decoder_hidden_params[0][:, ind] for ind in selection_bucket], dim=1),
                                     backend.stack([decoder_hidden_params[1][:, ind] for ind in selection_bucket], dim=1))
            if random.random() < self.teacher_forcing_ratio and target_variable is not None:
                decoder_input = row_wise_batch_copy(target_variable[di], k)  # Teacher forcing
            else:
                decoder_input = next_decoder_input
            target_length += 1
        if target_variable is not None:
            return loss, target_length, result.ids
        else:
            return result.log_probability, target_length, result.ids

    def optimizable_params_list(self) -> List[Any]:
        return [self.encoder.parameters(), self.decoder.parameters(), self.generator.parameters()]

    def validate_instance(self, source_tensor: backend.Tensor, reference_tensor: backend.Tensor) \
            -> Tuple[float, float, str]:
        """
        :param source_tensor: the input batch over which the predictions are generated
        :param reference_tensor: the expected Batch of sequences of ids
        :return: the bleu score between the reference and prediction batches, in addition to a sample result
        """
        with backend.no_grad():
            encoder_outputs, encoder_hidden_params = self.encode(source_tensor)
            if self.beam_size == 1:
                log_prob, target_length, predicted_ids = self.greedy_decode(encoder_outputs, encoder_hidden_params,None)
            else:
                log_prob, target_length, predicted_ids = self.beam_search_decode(encoder_outputs, encoder_hidden_params,
                                                                                 None, self.beam_size)
        bleu_score, ref_sample, hyp_sample = self.dataset.compute_bleu(
            reference_tensor, predicted_ids, ref_is_tensor=True, reader_level=self.dataset.get_target_word_granularity())
        result_sample = u"E=\"{}\", P=\"{}\"\n".format(ref_sample, hyp_sample)
        return bleu_score, log_prob, result_sample

    def update_model_parameters(self, args: Dict):
        """
        :param args: is expected to contain at least the two parameters "epoch" and "total"
        """
        if self.auto_teacher_forcing_ratio:
            self.teacher_forcing_ratio = 1.1 - float(args["epoch"] / args["total"])

