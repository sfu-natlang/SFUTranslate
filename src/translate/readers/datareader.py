"""
Provides the general dataset functionalities in the abstract class  :type AbsDatasetReader:
To create your own dataset reader you only need to extend this class and augment it with the functionalities
 you might need. :type DummyDataset: and :type ParallelTextDataset: are example dataset readers which extend this
  abstract class.
"""
from abc import ABC, abstractmethod
from typing import Callable, Iterable, Tuple, Dict, Iterator
from sacrebleu import sentence_bleu
from random import choice
from collections import Counter

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE

from translate.logging.utils import logger
from translate.readers.constants import ReaderLevel, ReaderType
from translate.readers.vocabulary import Vocab
from translate.configs.loader import ConfigLoader
from translate.configs.utils import Path
from translate.backend.utils import tensor2list

__author__ = "Hassan S. Shavarani"


class AbsDatasetReader(ABC):
    """
    The abstract interface of dataset readers, intended for reading a dataset, converting its data to NN understandable
    format (still in python structures tho!) and providing an iterator over the parallel data.
    """

    def __init__(self, configs: ConfigLoader, reader_type: ReaderType, shared_reader_data: Dict = None):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param reader_type: an intance of ReaderType enum stating the type of the dataste (e.g. Train, Test, Dev)
        :param shared_reader_data: the data shared from another reader to this reader instance
        """
        super(AbsDatasetReader, self).__init__()
        logger.info("Loading the dataset reader of type \"{}.{}\"".format(self.__class__.__name__, reader_type.name))
        self._iter_log_handler = None
        self.reader_type = reader_type
        self.configs = configs

        # _version will be used in logging and dumping of internal variables (e.g. vocabularies)
        self._version_ = 0.1
        # buffer size will be used in case multiple lines of the dataset is supposed to be buffered (and shuffled)
        self._instance_buffer_size = configs.get("reader.dataset.buffer_size", must_exist=True)
        # maximum valid sentence length for the model, in case of BPE-level ~ 128, in case of word-level ~ 50-60
        self._max_valid_length = configs.get("reader.dataset.max_length", must_exist=True)
        # the source side vocabulary data (only the container need to be filled in the classes extending the reader!)
        self.source_vocabulary = Vocab(configs)
        # the target side vocabulary data (only the container need to be filled in the classes extending the reader!)
        self.target_vocabulary = Vocab(configs)
        # the bpe_model instance which can get loaded with source train data
        self._source_bpe_model = None
        # the bpe_model instance which can get loaded with target train data
        self._target_bpe_model = None
        self.load_shared_reader_data(shared_reader_data)
        # setting the default token granularities
        self._src_word_granularity = ReaderLevel.WORD
        self._tgt_word_granularity = ReaderLevel.WORD

    def set_iter_log_handler(self, iter_log_handler: Callable[[str], None]):
        """
        :param iter_log_handler: the handler pointer of set_description handler of tqdm instance, iterating over this
         dataset. This handler is used to inform the user the progress of preparing the data while processing the
          dataset (which could sometimes take a long time). You are not forced to use it if you don't feel your dataset
           takes any time for data preparation. 
        """
        self._iter_log_handler = iter_log_handler

    @staticmethod
    def _sentensify(vocabulary: Vocab, ids: Iterable[int], input_is_tensor=False, reader_level: ReaderLevel=ReaderLevel.WORD):
        """
        The function which receives a list of ids, looks up each one in the dictionary and converts it back to its
         equivalent word. The input can be either a python list of integer ids or a one dimensional tensor of ids (
           input_is_tensor will indicate which one is the case). The function will also consider merging back word-piece
            (bpe) tokens together to help correct computation of BLEU score for the result. The :param reader_level: 
             will help performing this conversion from the token id level to the actual words in the sentence. 
        :return: equivalent sentence containing actual words in vocabulary, the format will be :type str:
        """
        if input_is_tensor:
            ids = tensor2list(ids)
        out_sent = " ".join([x for x in [vocabulary[x] for x in ids]
                             if x != vocabulary.pad_word and x != vocabulary.eos_word])

        if reader_level == ReaderLevel.BPE:
            bpe_separator = vocabulary.bpe_separator
            return out_sent.replace(" {}".format(bpe_separator), "").replace("{} ".format(bpe_separator), "")
        elif reader_level == ReaderLevel.CHAR:
            space_token = vocabulary.space_word
            return "".join(out_sent.split()).replace(space_token, " ")
        else:
            return out_sent

    def target_sentensify(self, ids: Iterable[int], input_is_tensor=False, reader_level: ReaderLevel=ReaderLevel.WORD):
        """
        receives a list(or tensor) of ids and converts it back to an string containing actual dictionary words.
         calls the internal static :method _sentensify: function (look at it's description for more details). 
          The :param reader_level: will help the internal functions perform the correct conversion from the 
            token id level to the actual words in the sentence.
        :return: equivalent sentence containing actual words in vocabulary, the format will be :type str:
        """
        return self._sentensify(self.target_vocabulary, ids, input_is_tensor, reader_level=reader_level)

    def target_sentensify_all(self, ids_list: Iterable[Iterable[int]], input_is_tensor=False,
                              reader_level: ReaderLevel=ReaderLevel.WORD):
        """
        receives a list of list(or tensor) of ids and converts them all back to strings containing actual dictionary
         words. calls the internal static :method _sentensify: function (look at it's description for more details) for
          each of them and returns the resulting strings containing actual words in vocabulary as a List[str]. 
           The :param reader_level: will help the internal functions perform the correct conversion from the 
            token id level to the actual words in the sentence.
        """
        return [self._sentensify(self.target_vocabulary, ids, input_is_tensor, reader_level=reader_level)
                for ids in ids_list]

    def compute_bleu(self, ref_ids_list: Iterable[Iterable[int]], hyp_ids_list: Iterable[Iterable[int]],
                     ref_is_tensor: bool = False, hyp_is_tensor: bool = False,
                     reader_level: ReaderLevel=ReaderLevel.WORD) -> Tuple[float, str, str]:
        """
        The wrapper function over sacrebleu.sentence_bleu which computes average bleu score over a pack of predicted
         sentences considering their equivalent single reference sentences. The input reference/prediction id lists can
          be either python lists or tensors (the flags :param ref_is_tensor: and :param hyp_is_tensor: indicate which is
           the case). The :param reader_level: will help the internal functions perform the correct conversion from the 
            token id level to the actual words in the sentence.
        :return: the computed average bleu score plus a sample pair of reference/prediction sentences which can be used
         for logging purposes (or totally ignored!)
        """
        assert len(ref_ids_list) == len(hyp_ids_list)
        refs = self.target_sentensify_all(ref_ids_list, input_is_tensor=ref_is_tensor, reader_level=reader_level)
        hyps = self.target_sentensify_all(hyp_ids_list, input_is_tensor=hyp_is_tensor, reader_level=reader_level)
        scores = [sentence_bleu(hyps[sid], refs[sid]) for sid in range(len(ref_ids_list))]
        random_index = choice(range(len(refs)))
        return sum(scores) / len(scores), refs[random_index], hyps[random_index]

    @staticmethod
    def retrieve_bpe_model(train_file: Path, merge_size: int, bpe_separator: str) -> BPE:
        """
        Given the train_file address (:param train_file:), the method checks the existence of it and creates the
         bpe_file with (:param merge_size:) number of merge operations and returns the address of the file
          the output model (say named "bpe") can simply segment a line using this command: bpe.segment(line)
           returning an object of str itself
        """
        base = train_file.stem  # the bare name of the file
        ext = train_file.suffix  # the suffix of the file starting with "."
        bpe_file = Path(train_file.parent / "{}.{}.bpe{}".format(base, merge_size, ext))
        if not bpe_file.exists():
            logger.info("Learning bpe merge operations for {}".format(ext[1:]))
            learn_bpe(train_file.open(encoding="utf-8"), bpe_file.open(mode='w', encoding='utf-8'), merge_size,
                      min_frequency=1, verbose=False, is_dict=False)
        return BPE(bpe_file.open(encoding='utf-8'), separator=bpe_separator)

    @staticmethod
    def load_vocab_counts(lines_stream: Iterator[str], vocab_counts_file: Path, min_count: int = 1):
        """
        The method to take a :param vocab_counts_file: to load the vocabulary from (the words above :param min_count:
         will get loaded from the file). The method will create the file if it does not exist by going through the
          actual resource senteces provided through :param lines_stream: to collect the vocabulary.
           The final vocab_counts_file would look like the following:
            word1<space>integer_number\n
            word2<space>integer_number\n
            word3<space>integer_number\n
            ...
        """
        if not vocab_counts_file.exists():
            vocab_counts_file.touch(mode=0o666)
            vocab_counts = Counter()
            lines_count = 0
            for line in lines_stream:
                lines_count += 1
                for word in line.strip().split():
                    vocab_counts[word] += 1
            with vocab_counts_file.open(mode='w') as vcf:
                for word, count in vocab_counts.most_common():
                    vcf.write("{} {}\n".format(word, count))
        result = []
        with vocab_counts_file.open(encoding="utf-8") as existing_vocab_file:
            for line in existing_vocab_file:
                line_parts = line.split()
                word = line_parts[0]
                count = int(line_parts[1])
                if count > min_count:
                    result.append(word)
        return result

    def __iter__(self):
        return self

    def get_sorce_token_granularity(self) -> ReaderLevel:
        return self._src_word_granularity

    def get_target_word_granularity(self) -> ReaderLevel:
        return self._tgt_word_granularity

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
    def max_sentence_length(self):
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

    @property
    @abstractmethod
    def instance_schema(self):
        """
        The class method which explicitly declares the exact types of each part of instance which gets omitted.
         The exact type will be used when padding and converting to Tensors, It also implecitely declares the number of 
          parts each instance would contain, e.g. the number would be equal to 1 for language modelling, would be 2 for 
           normal sequence to sequence, ...
        """
        raise NotImplementedError


class DatasetStats:
    """
    This class will collect the instance size information and can provide the maximum, minimum and average line length
     of the dataset. The class will be mainly used for logging purposes.
    """
    def __init__(self):
        self._min_size = float("inf")
        self._max_size = 0.0
        self.instances_count = 0.0
        self.instances_size = 0.0

    def update(self, instance_size):
        if instance_size < self.min_size:
            self._min_size = instance_size
        elif instance_size > self.max_size:
            self._max_size = instance_size
        self.instances_size += instance_size
        self.instances_count += 1.0

    @property
    def min_size(self):
        return self._min_size

    @property
    def max_size(self):
        return self._max_size

    @property
    def avg_size(self):
        return self.instances_size / (self.instances_count + 1e-32)
