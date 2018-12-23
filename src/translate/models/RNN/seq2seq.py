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
##################################################
"""
import random
from typing import List, Any, Tuple

from translate.configs.loader import ConfigLoader
from translate.models.RNN.decoder import DecoderRNN
from translate.models.RNN.encoder import EncoderRNN
from translate.models.backend.utils import backend, zeros_tensor, Variable, list_to_long_tensor, long_tensor
from translate.readers.datareader import AbsDatasetReader
from translate.models.abs.modelling import AbsCompleteModel
__author__ = "Hassan S. Shavarani"


class SequenceToSequence(AbsCompleteModel):
    def __init__(self, configs: ConfigLoader, train_dataset: AbsDatasetReader):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param train_dataset: the dataset from which the statistics regarding dataset will be looked up during
         model configuration
        """
        super(SequenceToSequence, self).__init__(backend.nn.NLLLoss())
        self.teacher_forcing_ratio = configs.get("trainer.model.tfr", 1.1)
        self.bidirectional_encoding = configs.get("trainer.model.bienc", True)
        hidden_size = configs.get("trainer.model.hsize", must_exist=True)
        n_e_layers = configs.get("trainer.model.nelayers", 1)
        n_d_layers = configs.get("trainer.model.ndlayers", 1)
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
        self.decoder = DecoderRNN(hidden_size, len(train_dataset.target_vocabulary), self.bidirectional_encoding,
                                  self.max_length, n_d_layers, self.batch_size, decoder_dropout)
        self.encoder_output_size = self.encoder.hidden_size
        if self.bidirectional_encoding:
            self.encoder_output_size *= 2
        for p in self.encoder.parameters():
            p.data.uniform_(-init_val, init_val)
        for p in self.decoder.parameters():
            p.data.uniform_(-init_val, init_val)

    def forward(self, input_variable: backend.Tensor, target_variable: backend.Tensor, *args, **kwargs) \
            -> Tuple[backend.Tensor, int, List[Any]]:

        batch_size = input_variable.size()[0]
        encoder_hidden = self.encoder.init_hidden(batch_size=batch_size)

        input_variable = Variable(input_variable.transpose(0, 1))
        target_variable = Variable(target_variable.transpose(0, 1))

        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]

        encoder_outputs = Variable(zeros_tensor(self.max_length, batch_size, self.encoder_output_size))
        encoder_outputs = encoder_outputs.cuda() if self.use_cuda else encoder_outputs

        loss = 0
        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_variable[ei], encoder_hidden, batch_size=batch_size)
            encoder_outputs[ei] = encoder_output[0]

        decoder_input = Variable(list_to_long_tensor([self.sos_token_id] * batch_size))
        decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

        decoder_hidden = self.decoder.init_hidden(batch_size=batch_size)

        output = long_tensor(target_length, batch_size, 1).squeeze(-1)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = \
                self.decoder(decoder_input, decoder_hidden, encoder_outputs, batch_size=batch_size)
            loss += self.criterion(decoder_output, target_variable[di])
            _, topi = decoder_output.data.topk(1)
            output[di] = Variable(topi.view(-1))
            if random.random() < self.teacher_forcing_ratio:
                decoder_input = target_variable[di]  # Teacher forcing
            else:
                decoder_input = Variable(topi.view(-1))
                decoder_input = decoder_input.cuda() if self.use_cuda else decoder_input

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

        return loss, target_length, result_decoded_word_ids

    def optimizable_params_list(self) -> List[Any]:
        return [self.encoder.parameters(), self.decoder.parameters()]
