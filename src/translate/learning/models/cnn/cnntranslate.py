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
import random
from typing import List, Any, Tuple

from translate.backend.utils import backend, zeros_tensor, Variable, list_to_long_tensor, long_tensor
from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
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
        super(ByteNet, self).__init__(backend.nn.CrossEntropyLoss(size_average=True)  # ignore_index=padding_index
        self.dataset = train_dataset
        self.batch_size = configs.get("trainer.model.bsize", must_exist=True)
        init_val = configs.get("trainer.model.init_val", 0.01)

        self.max_length = train_dataset.max_sentence_length()
        self.sos_token_id = train_dataset.target_vocabulary.get_begin_word_index()
        self.eos_token_id = train_dataset.target_vocabulary.get_end_word_index()
        self.pad_token_id = train_dataset.target_vocabulary.get_pad_word_index()
        self.use_cuda = backend.cuda.is_available()

    def forward(self, input_tensor: backend.Tensor, target_tensor: backend.Tensor, *args, **kwargs) \
            -> Tuple[backend.Tensor, int, List[Any]]:
        raise NotImplementedError

    @abstractmethod
    def optimizable_params_list(self) -> List[Any]:
        raise NotImplementedError

    @abstractmethod
    def validate_instance(self, *args, **kwargs) -> Tuple[float, float, str]:
        raise NotImplementedError
