"""
The RMM language model module with an additional functionality to compute the loss value in the forward path
 (the loss is added to module for speed optimization purposes). The module can be configured via setting the following
  values in the config file to the desired values.
##################################################
trainer:
    model:
        type: rnnlm
        tfr: 1.001 # teacher forcing ratio
        bienc: true # bidirectional encoding
        hsize: 256 # hidden state size of RNN layers
        nelayers: 1 # number of hidden layers in encoder
        ndlayers: 1 # number of hidden layers in decoder
        bsize: 64 # size of the training sentence batches
        ddropout: 0.1 # the dropout probability in the decoder
        init_val: 0.1 # the value to range of which random variables get initiated
##################################################
"""
import math
from random import choice
from typing import List, Any, Tuple, Iterable

from translate.backend.utils import backend, long_tensor, zeros_tensor
from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
from translate.learning.modules.mlp.generator import GeneratorNN
from translate.readers.datareader import AbsDatasetReader

__author__ = "Hassan S. Shavarani"


class RNNLM(AbsCompleteModel):
    def __init__(self, configs: ConfigLoader, train_dataset: AbsDatasetReader):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param train_dataset: the dataset from which the statistics regarding dataset will be looked up during
         model configuration
        """
        super(RNNLM, self).__init__(backend.nn.NLLLoss())
        self.dataset = train_dataset
        self.teacher_forcing_ratio = configs.get("trainer.model.tfr", 1.1)
        self.bidirectional_encoding = configs.get("trainer.model.bienc", True)
        hidden_size = configs.get("trainer.model.hsize", must_exist=True)
        n_e_layers = configs.get("trainer.model.nelayers", 1)
        decoder_dropout = configs.get("trainer.model.ddropout", 0.1)
        init_val = configs.get("trainer.model.init_val", 0.01)
        self.batch_size = configs.get("trainer.model.bsize", must_exist=True)

        self.max_length = train_dataset.max_sentence_length()
        self.sos_token_id = train_dataset.target_vocabulary.get_begin_word_index()
        self.eos_token_id = train_dataset.target_vocabulary.get_end_word_index()
        self.pad_token_id = train_dataset.target_vocabulary.get_pad_word_index()
        self.enc_embedding = backend.nn.Embedding(len(train_dataset.source_vocabulary), hidden_size)
        self.enc_lstm = backend.nn.LSTM(hidden_size, hidden_size, bidirectional=self.bidirectional_encoding,
                                        num_layers=n_e_layers)
        self.num_enc_layers = n_e_layers
        self.enc_hidden_size = hidden_size
        self.encoder_output_size = hidden_size
        if self.bidirectional_encoding:
            self.encoder_output_size *= 2
        self.generator = GeneratorNN(self.encoder_output_size, len(train_dataset.target_vocabulary), decoder_dropout)
        for p_set in self.optimizable_params_list():
            for p in p_set:
                p.data.uniform_(-init_val, init_val)

    def forward(self, input_variable: backend.Tensor, *args, **kwargs) -> Tuple[backend.Tensor, int, List[Any]]:

        n_dirs = 2 if self.bidirectional_encoding else 1
        batch_size = input_variable.size(0)
        input_variable = input_variable.transpose(0, 1)
        input_length = input_variable.size(0)
        hidden_layer_params = zeros_tensor(n_dirs * self.num_enc_layers, batch_size, self.enc_hidden_size), \
            zeros_tensor(n_dirs * self.num_enc_layers, batch_size, self.enc_hidden_size)
        embedded_input = self.enc_embedding(input_variable)
        output = long_tensor(input_length, batch_size, 1).squeeze(-1)
        loss = 0
        for ei in range(input_length - 1):
            encoder_input = embedded_input[ei].view(1, batch_size, self.enc_hidden_size)
            encoder_output, hidden_layer_params = self.enc_lstm(encoder_input, hidden_layer_params)
            lm_output = self.generator(encoder_output).squeeze(0)
            loss += self.criterion(lm_output, input_variable[ei + 1])
            _, topi = lm_output.data.topk(1)
            output[ei] = topi.view(-1).detach()

        output = output.transpose(0, 1)
        result_decoded_word_ids = []
        for di in range(output.size()[0]):
            sent = []
            for word in output[di]:
                word = word.item()
                if word != self.pad_token_id:
                    sent.append(word)
                if word == self.eos_token_id:
                    break
            result_decoded_word_ids.append(sent)
        return loss, input_length, result_decoded_word_ids

    def optimizable_params_list(self) -> List[Any]:
        return [self.enc_embedding.parameters(), self.enc_lstm.parameters(), self.generator.parameters()]

    def validate_instance(self, prediction_loss: float, predictions_batch: Iterable[Iterable[int]],
                          input_batch: backend.Tensor) -> Tuple[float, float, str]:
        hyps = self.dataset.target_sentensify_all(
            predictions_batch, reader_level=self.dataset.get_target_word_granularity())
        random_index = choice(range(len(hyps)))
        return math.exp(prediction_loss), prediction_loss, hyps[random_index]
