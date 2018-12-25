"""
Provides the general dataset functionalities in the abstract class  :type AbsDatasetReader:
To create your own dataset reader you only need to extend this class and augment it with the functionalities
 you might need. :type DummyDataset: and :type ParallelTextDataset: are example dataset readers which extend this
  abstract class.
"""
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Tuple, Dict
from sacrebleu import sentence_bleu
from random import choice

from translate.logging.utils import logger
from translate.readers.constants import ReaderLevel, ReaderType
from translate.readers.vocabulary import Vocab
from translate.configs.loader import ConfigLoader
from translate.backend.utils import tensor2list

__author__ = "Hassan S. Shavarani"


class AbsDatasetReader(ABC):
    """
    The abstract interface of dataset readers, intended for reading a dataset, converting its data to NN understandable
    format (still in python structures tho!) and providing an iterator over the parallel data.
    """

    def __init__(self, configs: ConfigLoader, reader_type: ReaderType, iter_log_handler: Callable[[str], None] = None,
                 shared_reader_data: Dict = None):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param reader_type: an intance of ReaderType enum stating the type of the dataste (e.g. Train, Test, Dev)
        :param iter_log_handler: the handler pointer of set_description handler of tqdm instance, iterating over this
         dataset. This handler is used to inform the user the progress of preparing the data while processing the
          dataset (which could sometimes take a long time). You are not forced to use it if you don't feel your dataset
           takes any time for data preparation.
        :param shared_reader_data: the data shared from another reader to this reader instance
        """
        super(AbsDatasetReader, self).__init__()
        logger.info("Loading the dataset reader of type \"{}.{}\"".format(self.__class__.__name__, reader_type.name))
        self._iter_log_handler = iter_log_handler
        self.reader_type = reader_type
        self.configs = configs

        # _version will be used in logging and dumping of internal variables (e.g. vocabularies)
        self._version_ = 0.1
        # buffer size will be used in case multiple lines of the dataset is supposed to be buffered (and shuffled)
        self._instance_buffer_size = configs.get("reader.dataset.buffer_size", must_exist=True)
        # maximum valid sentence length for the model, in case of BPE-level ~ 128, in case of word-level ~ 50-60
        self._max_valid_length = configs.get("reader.dataset.max_length", must_exist=True)
        # word granularity can be used in the dataset reader to prepare the data in a specific format
        self._word_granularity = configs.get("reader.dataset.granularity", default_value=ReaderLevel.WORD)
        # the source side vocabulary data (only the container need to be filled in the classes extending the reader!)
        self.source_vocabulary = Vocab(configs)
        # the target side vocabulary data (only the container need to be filled in the classes extending the reader!)
        self.target_vocabulary = Vocab(configs)
        self.load_shared_reader_data(shared_reader_data)

    @staticmethod
    def _sentensify(vocabulary: Vocab, ids: Iterable[int], merge_bpe_tokens: bool = False, input_is_tensor=False):
        """
        The function which receives a list of ids, looks up each one in the dictionary and converts it back to its
         equivalent word. The input can be either a python list of integer ids or a one dimensional tensor of ids (
           input_is_tensor will indicate which one is the case). The function will also consider merging back word-piece
            (bpe) tokens together (if merge_bpe_tokens is true) to help correct computation of BLEU score for the result
        :return: equivalent sentence containing actual words in vocabulary, the format will be :type str:
        """
        if input_is_tensor:
            ids = tensor2list(ids)
        out_sent = " ".join([x for x in [vocabulary[x] for x in ids]
                             if x != vocabulary.pad_word and x != vocabulary.eos_word])
        if not merge_bpe_tokens:
            return out_sent
        else:
            bpe_separator = vocabulary.bpe_separator
            return out_sent.replace(" {}".format(bpe_separator), "").replace("{} ".format(bpe_separator), "")

    def target_sentensify(self, ids: Iterable[int], merge_bpe_tokens: bool = False, input_is_tensor=False):
        """
        receives a list(or tensor) of ids and converts it back to an string containing actual dictionary words.
        calls the internal static :method _sentensify: function (look at it's description for more details).
        :return: equivalent sentence containing actual words in vocabulary, the format will be :type str:
        """
        return self._sentensify(self.target_vocabulary, ids, merge_bpe_tokens, input_is_tensor)

    def target_sentensify_all(self, ids_list: Iterable[Iterable[int]],
                              merge_bpe_tokens: bool = False, input_is_tensor=False):
        """
        receives a list of list(or tensor) of ids and converts them all back to strings containing actual dictionary
         words. calls the internal static :method _sentensify: function (look at it's description for more details) for
          each of them and returns the resulting strings containing actual words in vocabulary as a List[str].
        """
        return [self._sentensify(self.target_vocabulary, ids, merge_bpe_tokens, input_is_tensor) for ids in ids_list]

    def compute_bleu(self, ref_ids_list: Iterable[Iterable[int]], hyp_ids_list: Iterable[Iterable[int]],
                     ref_is_tensor: bool = False, hyp_is_tensor: bool = False) -> Tuple[float, str, str]:
        """
        The wrapper function over sacrebleu.sentence_bleu which computes average bleu score over a pack of predicted
         sentences considering their equivalent single reference sentences. The input reference/prediction id lists can
          be either python lists or tensors (the flags :param ref_is_tensor: and :param hyp_is_tensor: indicate which is
           the case).
        :return: the computed average bleu score plus a sample pair of reference/prediction sentences which can be used
         for logging purposes (or totally ignored!)
        """
        assert len(ref_ids_list) == len(hyp_ids_list)
        refs = self.target_sentensify_all(ref_ids_list, input_is_tensor=ref_is_tensor)
        hyps = self.target_sentensify_all(hyp_ids_list, input_is_tensor=hyp_is_tensor)
        scores = [sentence_bleu(hyps[sid], refs[sid]) for sid in range(len(ref_ids_list))]
        random_index = choice(range(len(refs)))
        return sum(scores) / len(scores), refs[random_index], hyps[random_index]

    def max_sentence_length(self):
        return self._max_valid_length

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        """
        The most important function in the reader is this function. This function is in charge of looking at the
         provided resource to itself and reads on single instance from it (e.g. a pair of French-English sentences from
          the train dataset text file), converts them to NN understandable format (e.g. converts them to a pair of id
           lists using the filled language vocabulary instances) and returns them. Please pay attention that this method
            is by no means in charge of batching the data. Batching, padding and all the other post processing steps
             would need to be taken somewhere else (presumably in a backend related class in models package!)
        :return: a pair of un-padded id lists
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        :return: the total size of instances in the dataset (not just the buffered instances)
        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        """
        :return: the instance in index :param idx: of the dataset (can be simply calling next but with caution that the
         element in position idx can be different each time one accesses it!)
        """
        raise NotImplementedError

    @abstractmethod
    def allocate(self):
        """
        the function which allocates the resources for the dataset reader to go over them (e.g. opens files,
         loades the tokenizers, ...). Please pay attention that this function should not be mistaken with init function,
           since this function may be called once (multiple times) every epoch during the training.
        :return: nothing
        """
        raise NotImplementedError

    @abstractmethod
    def deallocate(self):
        """
        the function which de-allocates the resources for the dataset reader to go over them (e.g. closes files,
         unloades the tokenizers, ...)
        :return: nothing
        """
        raise NotImplementedError

    @abstractmethod
    def load_shared_reader_data(self, shared_data):
        """
        The class method in charge of loading the external :param shared_data: information into reader instance. This 
         method is useful in sharing vocabulary and other shared data among different parts (TRAIN, TEST, DEV) of the 
          same dataset
        """
        raise NotImplementedError

    @abstractmethod
    def get_sharable_data(self):
        """
        The class method to provide useful trained data (e.g. vocabulary objects) mainly from TRAIN dataset reader to 
         the TEST and DEV dataset readers.
        """
        raise NotImplementedError
