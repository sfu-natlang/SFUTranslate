from abc import ABC, abstractmethod
from typing import Callable

from translate.readers.constants import ReaderLevel, ReaderType
from translate.readers.vocabulary import Vocab
from translate.configs.loader import ConfigLoader


class AbsDatasetReader(ABC):
    """
    The abstract interface of dataset readers, intended for reading a dataset, converting its data to NN understandable
    format and providing an iterator over the parallel data
    """
    def __init__(self, configs: ConfigLoader, reader_type: ReaderType, iter_log_handler: Callable[[str], None] = None):
        super(AbsDatasetReader, self).__init__()
        self.iter_log_handler = iter_log_handler
        self.reader_type = reader_type
        self.configs = configs
        self._version_ = 0.1
        self.instance_buffer_size = configs.get("reader.dataset.buffer_size", must_exist=True)
        self.max_valid_length = configs.get("reader.dataset.max_length", must_exist=True)
        self.word_granularity = configs.get("reader.dataset.granularity", default_value=ReaderLevel.WORD)
        self.vocabulary = Vocab(configs)

    @abstractmethod
    def __next__(self):
        raise NotImplementedError
