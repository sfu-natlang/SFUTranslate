"""
The cnn translate module based on the paper "Neural Machine Translation in Linear Time"
 (https://arxiv.org/abs/1610.10099) and using the suggested implementation in
  (https://github.com/dhpollack/bytenet.pytorch) with an additional functionality to compute the loss value in the
   forward path (the loss is added to module for speed optimization purposes). The module can be configured via
    setting the following values in the config file to the desired values.
##################################################
trainer:
    model:
        type: bytenet
        bsize: 64 # size of the training sentence batches
        init_val: 0.1 # the value to range of which random variables get initiated
        d: 400 # number of features in network
        max_r: 16 # max dilation size
        n_sets: 6 # number of ResBlock sets
        k: 3 # kernel size
##################################################
"""
from typing import List, Any, Tuple

from translate.backend.utils import backend
from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
from translate.learning.modules.cnn.decoder import CharCNNDecoder
from translate.learning.modules.cnn.encoder import CharCNNEncoder
from translate.readers.datareader import AbsDatasetReader
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


class ByteNet(AbsCompleteModel):
    def __init__(self, configs: ConfigLoader, train_dataset: AbsDatasetReader):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param train_dataset: the dataset from which the statistics regarding dataset will be looked up during
         model configuration
        """
        super(ByteNet, self).__init__(backend.nn.CrossEntropyLoss(reduction='elementwise_mean'))
        # ignore_index=padding_index
        self.dataset = train_dataset
        self.batch_size = configs.get("trainer.model.bsize", must_exist=True)
        init_val = configs.get("trainer.model.init_val", 0.01)
        input_features = configs.get("trainer.model.d", must_exist=True)
        max_r = configs.get("trainer.model.max_r", must_exist=True)
        num_sets = configs.get("trainer.model.n_sets", must_exist=True)
        k = configs.get("trainer.model.k", must_exist=True)

        self.max_length = train_dataset.max_sentence_length()
        self.sos_token_id = train_dataset.target_vocabulary.get_begin_word_index()
        self.eos_token_id = train_dataset.target_vocabulary.get_end_word_index()
        self.tgt_pad_token_id = train_dataset.target_vocabulary.get_pad_word_index()
        self.src_pad_token_id = train_dataset.source_vocabulary.get_pad_word_index()
        self.use_cuda = backend.cuda.is_available()

        self.encoder = CharCNNEncoder(input_features // 2, max_r, k, num_sets)
        self.decoder = CharCNNDecoder(input_features // 2, max_r, k, num_sets,
                                      len(train_dataset.target_vocabulary), use_logsm=False)

        logger.info("Randomly initiating model variables in the range [-{0}, {0}]".format(init_val))
        for p_set in self.optimizable_params_list():
            for p in p_set:
                p.data.uniform_(-init_val, init_val)

    def forward(self, input_tensor: backend.Tensor, target_tensor: backend.Tensor, *args, **kwargs) \
            -> Tuple[backend.Tensor, int, List[Any]]:
        input_tensor, target_tensor = self.equalize_tensor_lengths(input_tensor, target_tensor)
        out = self.decoder(self.encoder(input_tensor.unsqueeze(1).float()))
        loss = self.criterion(out, target_tensor)
        return loss, (input_tensor != self.tgt_pad_token_id).sum().item(), []

    def optimizable_params_list(self) -> List[Any]:
        return [self.encoder.parameters(), self.decoder.parameters()]

    def validate_instance(self, prediction_loss: float, hyp_ids_list: List[List[int]], input_id_list: backend.Tensor,
                          ref_ids_list: backend.Tensor, *args, **kwargs) -> Tuple[float, float, str]:
        """
        :param prediction_loss: the model calculated loss value over the current prediction
        :param hyp_ids_list: the current predicted Batch of sequences of ids. In case of this model the value is always
         an empty list (it has not been removed from the api to make the interface consistent with the other types of
         model). This value can be safely ignored for this model since it will get computed inside this function
        :param input_id_list: the input batch over which the predictions are generated
        :param ref_ids_list: the expected Batch of sequences of ids
        :param args: contains the Transformer style mask tensors (as the last two indices of the args list)
        :return: the bleu score between the reference and prediction batches, in addition to a sample result
        """
        hyp_ids_list = []
        hyp_ids_tensor = self.decoder(self.encoder(input_id_list.unsqueeze(1).float())).argmax(dim=1)
        for sentence_index in range(hyp_ids_tensor.size(0)):
            sent = []
            for word in hyp_ids_tensor[sentence_index]:
                word = word.item()
                if word != self.tgt_pad_token_id:
                    sent.append(word)
                if word == self.eos_token_id:
                    break
            hyp_ids_list.append(sent)
        bleu_score, ref_sample, hyp_sample = self.dataset.compute_bleu(
            ref_ids_list[:, 1:], hyp_ids_list, ref_is_tensor=True,
            reader_level=self.dataset.get_target_word_granularity())
        result_sample = u"E=\"{}\", P=\"{}\"\n".format(ref_sample, hyp_sample)
        return bleu_score, prediction_loss, result_sample

    def equalize_tensor_lengths(self, input_tensor: backend.Tensor, target_tensor: backend.Tensor) \
            -> Tuple[backend.Tensor, backend.Tensor]:
        """
        The output of the bytenet is of the same size as the :param input_tensor: while the loss is calculated on
         the :param target_tensor: so this function makes sure the two are of the same length before getting passed
           through the ByteNet model.
        """
        i_size = input_tensor.size(-1)
        o_size = target_tensor.size(-1)
        if i_size < o_size:
            padds = backend.ones(list(input_tensor.size()[:-1]) + [o_size - i_size]).type_as(
                input_tensor.data) * self.src_pad_token_id
            input_tensor = backend.cat([input_tensor, padds], dim=-1)
        elif i_size > o_size:
            padds = backend.ones(list(target_tensor.size()[:-1]) + [i_size - o_size]).type_as(
                target_tensor.data) * self.tgt_pad_token_id
            target_tensor = backend.cat([target_tensor, padds], dim=-1)
        assert input_tensor.size(-1) == target_tensor.size(-1)
        return input_tensor, target_tensor
