"""
Provides dummy datasets for proof-of-concept tasks in NLP (e.g. reverse copy for NMT or language modelling for
 sequences generated using a simple Grammer). For further information please read through the docstrings in each class.
"""
import string
from random import randint, choice
from typing import Callable, Dict

from translate.configs.loader import ConfigLoader
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader

__author__ = "Hassan S. Shavarani"


class ReverseCopyDataset(AbsDatasetReader):
    """
    Provides a generated pack of sentences which mimics the task of copy + reverse in neural machine translation, e.g. for a
     sentence "a b c d e f a r" it should provide "r a f e d c b a" as the translation of the instance. To make your model
      use this reader, set "reader.dataset.type" in your config file to "dummy" and set the following values in it with your
        desired values:
    ####################################
        reader:
            dataset:
                type: dummy_s2s
                dummy:
                    min_len: 8
                    max_len: 50
                    vocab_size: 96
                    train_samples: 40000
                    test_samples: 3000
                    dev_samples: 1000
        trainer:
            experiment:
                name: 'dummy'
    ####################################
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
        super().__init__(configs, reader_type, iter_log_handler, shared_reader_data)
        self.min_length = configs.get("reader.dataset.dummy.min_len", must_exist=True)
        self.max_length = configs.get("reader.dataset.dummy.max_len", must_exist=True)
        if reader_type == ReaderType.TRAIN:
            self.max_samples = configs.get("reader.dataset.dummy.train_samples", must_exist=True)
        elif reader_type == ReaderType.TEST:
            self.max_samples = configs.get("reader.dataset.dummy.test_samples", must_exist=True)
        elif reader_type == ReaderType.DEV:
            self.max_samples = configs.get("reader.dataset.dummy.dev_samples", must_exist=True)
        else:
            self.max_samples = 1
        # The desired vocabulary given the vocab_size set in config file gets created in here
        #  and is set inside source and target vocabulry objects
        if reader_type == ReaderType.TRAIN:
            vocab_size = configs.get("reader.dataset.dummy.vocab_size", must_exist=True)
            tmp = [x for x in string.ascii_letters + string.punctuation + string.digits]
            vocab = [x + "," + y for x in tmp for y in tmp if x != y][:vocab_size]
            self.source_vocabulary.set_types(vocab)
            self.target_vocabulary.set_types(vocab)
        if self.max_samples > 1:
            self.pairs = [self._get_next_pair() for _ in range(self.max_samples)]
        else:
            self.pairs = [self._get_next_pair(self.max_length)]
        self.reading_index = 0

    def max_sentence_length(self):
        return self.max_length

    def __next__(self):
        """
        The function always iterates over the already generated/cached pairs of sequences (with their reverse sequence)
        """
        if self.reading_index < len(self.pairs):
            tmp = self.pairs[self.reading_index]
            self.reading_index += 1
            return tmp
        else:
            self.reading_index = 0
        raise StopIteration

    def __getitem__(self, idx):
        return self.pairs[idx]

    def __len__(self):
        return len(self.pairs)

    def allocate(self):
        return

    def deallocate(self):
        return

    def load_shared_reader_data(self, shared_data):
        if self.reader_type != ReaderType.TRAIN and shared_data is None:
            raise ValueError("Only trainer instance is allowed to create the vocabulary and dummy sentences!")
        if shared_data is not None:
            vocab = shared_data["types"]
            self.source_vocabulary.set_types(vocab)
            self.target_vocabulary.set_types(vocab)

    def get_sharable_data(self):
        return {"types": self.source_vocabulary.get_types()}

    def _get_next_pair(self, expected_length=0):
        """
        The function which given an :param expected_length: size of sentence, creates a random string from the
         vocabulary and returns it along with its reverse in return
        """
        if expected_length == 0:
            expected_length = randint(self.min_length - 1, self.max_length - 1)
        if expected_length >= self.max_length:
            expected_length = self.max_length - 1
        tmp = [self.target_vocabulary[x] for x in self.target_vocabulary.retrieve_dummy_words_list(expected_length)]
        rev = [x for x in tmp[::-1]]
        tmp += [self.target_vocabulary.get_end_word_index()]
        rev += [self.target_vocabulary.get_end_word_index()]
        return tmp, rev


class SimpleGrammerLMDataset(AbsDatasetReader):
    """
    Generates a set of sentences following the given grammer:
     1. pick a random word in dictionary as w_0
     2. for the rest of the words (w_i) up to a maximum length (randomly chosen between min_l and max_l)
         2.1 previous_word_index <- vocabulary index of word w_{i-1}
         2.2 for w_i, randomly select the word either in position (previous_word_index + 1) or (previous_word_index - 1)
     e.g. the following sequence can be an output of such grammer given a vocabulary of lower-case english letters:
         "d e f e d c d c d c"
     end for the next word prediction we expect the highest probability of the model to be on "b" or "d". 
     To make your model
      use this reader, set "reader.dataset.type" in your config file to "dummy" and set the following values in it with your
        desired values:
    ####################################
        reader:
            dataset:
                type: dummy_lm
                dummy:
                    min_len: 8
                    max_len: 50
                    vocab_size: 96
                    train_samples: 40000
                    test_samples: 3000
                    dev_samples: 1000
        trainer:
            experiment:
                name: 'dummy'
    ####################################
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
        super().__init__(configs, reader_type, iter_log_handler, shared_reader_data)
        self.min_length = configs.get("reader.dataset.dummy.min_len", must_exist=True)
        self.max_length = configs.get("reader.dataset.dummy.max_len", must_exist=True)
        if reader_type == ReaderType.TRAIN:
            self.max_samples = configs.get("reader.dataset.dummy.train_samples", must_exist=True)
        elif reader_type == ReaderType.TEST:
            self.max_samples = configs.get("reader.dataset.dummy.test_samples", must_exist=True)
        elif reader_type == ReaderType.DEV:
            self.max_samples = configs.get("reader.dataset.dummy.dev_samples", must_exist=True)
        else:
            self.max_samples = 1
        # The desired vocabulary given the vocab_size set in config file gets created in here
        #  and is set inside source and target vocabulry objects
        if reader_type == ReaderType.TRAIN:
            vocab_size = configs.get("reader.dataset.dummy.vocab_size", must_exist=True)
            tmp = [x for x in string.ascii_letters + string.punctuation + string.digits]
            vocab = [x + "," + y for x in tmp for y in tmp if x != y][:vocab_size]
            self.source_vocabulary.set_types(vocab)
            self.target_vocabulary.set_types(vocab)
        if self.max_samples > 1:
            self.sentences = [self._get_next_sentence() for _ in range(self.max_samples)]
        else:
            self.sentences = [self._get_next_sentence(self.max_length)]
        self.reading_index = 0

    def max_sentence_length(self):
        return self.max_length

    def __next__(self):
        """
        The function always iterates over the already generated/cached pairs of sequences (with their reverse sequence)
        """
        if self.reading_index < len(self.sentences):
            tmp = self.sentences[self.reading_index]
            self.reading_index += 1
            return tmp
        else:
            self.reading_index = 0
        raise StopIteration

    def __getitem__(self, idx):
        return self.sentences[idx]

    def __len__(self):
        return len(self.sentences)

    def allocate(self):
        return

    def deallocate(self):
        return

    def load_shared_reader_data(self, shared_data):
        if self.reader_type != ReaderType.TRAIN and shared_data is None:
            raise ValueError("Only trainer instance is allowed to create the vocabulary and dummy sentences!")
        if shared_data is not None:
            vocab = shared_data["types"]
            self.source_vocabulary.set_types(vocab)
            self.target_vocabulary.set_types(vocab)

    def get_sharable_data(self):
        return {"types": self.source_vocabulary.get_types()}

    def _get_next_sentence(self, expected_length=0):
        """
        The function which given an :param expected_length: size of sentence, draws a sentence from the defined grammar,
         and returns it
        """
        if expected_length == 0:
            expected_length = randint(self.min_length - 1, self.max_length - 1)
        if expected_length >= self.max_length:
            expected_length = self.max_length - 1
        vocab_length = len(self.target_vocabulary)
        next_index_increase = [-1, +1]
        actions = [choice(next_index_increase) for _ in range(expected_length - 1)]
        actions.insert(0, 0)
        first_word_index = choice(range(vocab_length))
        return [[(first_word_index + sum(actions[:i])) % vocab_length for i in range(1, len(actions) + 1)]]
