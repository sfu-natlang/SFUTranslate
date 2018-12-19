import string
from random import randint

from translate.configs.loader import ConfigLoader
from translate.readers.constants import ReaderType
from translate.readers.datareader import AbsDatasetReader


class DummyDataset(AbsDatasetReader):
    def __init__(self, configs: ConfigLoader, reader_type: ReaderType):
        super().__init__(configs, reader_type)
        self.min_length = configs.get("reader.dataset.dummy.min_len", must_exist=True)
        self.max_length = configs.get("reader.dataset.dummy.max_len", must_exist=True)
        self.vocab_size = configs.get("reader.dataset.dummy.vocab_size", must_exist=True)
        if reader_type == ReaderType.TRAIN:
            self.max_samples = configs.get("reader.dataset.dummy.train_samples", must_exist=True)
        elif reader_type == ReaderType.TEST:
            self.max_samples = configs.get("reader.dataset.dummy.test_samples", must_exist=True)
        elif reader_type == ReaderType.DEV:
            self.max_samples = configs.get("reader.dataset.dummy.dev_samples", must_exist=True)
        else:
            self.max_samples = 1
        tmp = [x for x in string.ascii_letters + string.punctuation + string.digits]
        vocab = [x + "," + y for x in tmp for y in tmp if x != y][:self.vocab_size]
        self.e_vocabulary.set_types(vocab)
        if self.max_samples > 1:
            self.pairs = [self.__get_next_pair() for _ in range(self.max_samples)]
        else:
            self.pairs = [self.__get_next_pair(self.max_length)]
        self.reading_index = 0

    def __next__(self):
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

    def __get_next_pair(self, expected_length=0):
        if expected_length == 0:
            expected_length = randint(self.min_length - 1, self.max_length - 1)
        if expected_length >= self.max_length:
            expected_length = self.max_length - 1
        tmp = [self.e_vocabulary[x] for x in self.e_vocabulary.retrieve_dummy_words_list(expected_length)]
        rev = [x for x in tmp[::-1]]
        tmp += [self.e_vocabulary.get_end_word_index()]
        rev += [self.e_vocabulary.get_end_word_index()]
        return tmp, rev
