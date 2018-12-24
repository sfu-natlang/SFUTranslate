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

from translate.configs.loader import ConfigLoader
from translate.models.RNN.encoder import EncoderRNN
from translate.models.RNN.generator import GeneratorNN
from translate.models.backend.utils import backend, Variable, long_tensor
from translate.readers.datareader import AbsDatasetReader
from translate.models.abs.modelling import AbsCompleteModel


__author__ = "Hassan S. Shavarani"


class RNNLM(AbsCompleteModel):
    def __init__(self, configs: ConfigLoader, train_dataset: AbsDatasetReader):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param train_dataset: the dataset from which the statistics regarding dataset will be looked up during
         model configuration
        """
        super(RNNLM, self).__init__(backend.nn.NLLLoss())
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
        self.use_cuda = backend.cuda.is_available()

        self.encoder = EncoderRNN(len(train_dataset.source_vocabulary),
                                  hidden_size, self.bidirectional_encoding, n_e_layers, self.batch_size)
        self.encoder_output_size = self.encoder.hidden_size
        if self.bidirectional_encoding:
            self.encoder_output_size *= 2
        self.generator = GeneratorNN(self.encoder_output_size, len(train_dataset.target_vocabulary), decoder_dropout)
        for p_set in self.optimizable_params_list():
            for p in p_set:
                p.data.uniform_(-init_val, init_val)

    def forward(self, input_variable: backend.Tensor, *args, **kwargs) -> Tuple[backend.Tensor, int, List[Any]]:

        batch_size = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)

        input_variable = Variable(input_variable.transpose(0, 1))

        input_length = input_variable.size()[0]

        output = long_tensor(input_length, batch_size, 1).squeeze(-1)

        loss = 0
        for ei in range(input_length - 1):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden, batch_size=batch_size)
            lm_output = self.generator(encoder_output).squeeze(0)
            loss += self.criterion(lm_output, input_variable[ei + 1])
            _, topi = lm_output.data.topk(1)
            output[ei] = Variable(topi.view(-1))

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
        return [self.encoder.parameters(), self.generator.parameters()]

    def validate_instance(self, prediction_loss: float, predictions_batch: Iterable[Iterable[int]],
                          input_batch: backend.Tensor) -> Tuple[float, float, str]:
        hyps = self.target_sentensify_all(predictions_batch)
        random_index = choice(range(len(hyps)))
        return math.exp(prediction_loss), prediction_loss, hyps[random_index]
