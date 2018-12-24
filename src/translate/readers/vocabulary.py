"""
Provides a data container for the vocabulary words of one language (either "e" or "f").
"""
from typing import List
from collections import Counter
from translate.configs.loader import ConfigLoader
from random import choice

__author__ = "Hassan S. Shavarani"


class Vocab:
    def __init__(self, configs: ConfigLoader, words: List[str] = None, counter: Counter = None):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param words: list of unique tokens in the vocabulary
        :param counter: [optional] counter containing the frequency of the words in the words list.
        """
        self.unk_word = configs.get("reader.vocab.unk_word", "<unk>")
        self.bos_word = configs.get("reader.vocab.bos_word", "<s>")
        self.eos_word = configs.get("reader.vocab.eos_word", "</s>")
        self.pad_word = configs.get("reader.vocab.pad_word", "<pad>")
        self.bpe_separator = configs.get("reader.vocab.bpe_separator", "@@")
        self.i2w = words
        self.w2i = None  # to be loaded lazily!
        self.counter = counter

    def set_types(self, words: List[str], counter: Counter = None):
        """
        The setter function which will set the vocabulary word list (:param words:) and the optional (:param counter:)
        """
        self.i2w = words
        self.counter = counter
        self.ensure_words_exist([self.bos_word, self.eos_word, self.pad_word, self.unk_word])
        self._fill_in_reverse_index_vocabulary()

    def __len__(self):
        return len(self.i2w)

    def get_types(self):
        return self.i2w

    def get_end_word_index(self):
        assert self.w2i is not None
        return self.w2i[self.eos_word]

    def get_begin_word_index(self):
        assert self.w2i is not None
        return self.w2i[self.bos_word]

    def get_pad_word_index(self):
        assert self.w2i is not None
        return self.w2i[self.pad_word]

    def get_unk_word_index(self):
        assert self.w2i is not None
        return self.w2i[self.unk_word]

    def ensure_words_exist(self, words: List[str]):
        assert self.i2w is not None
        for word in words:
            if word not in self.i2w:
                self.i2w.append(word)
                if self.w2i is not None:
                    self.w2i[word] = len(self.w2i)
                if self.counter is not None:
                    self.counter[word] = 1

    def _fill_in_reverse_index_vocabulary(self):
        assert self.i2w is not None
        del self.w2i
        self.w2i = {}
        for ind, word in enumerate(self.i2w):
            self.w2i[word] = ind

    def __getitem__(self, item):
        """
        The getter function which given an id or word returns the word or id equivalent to it
        :param item: The id or word to be converted to it's equivalent
        :return: The converted word or id equivalent of the :param item:
        """
        assert self.i2w is not None
        assert type(item) == int or type(item) == str
        if type(item) == int:
            if 0 <= item < len(self.i2w):
                return self.i2w[item]
            else:
                return self.pad_word
        else:
            if self.w2i is None:
                self._fill_in_reverse_index_vocabulary()
            if item in self.w2i:
                return self.w2i[item]
            else:
                return self.w2i[self.unk_word]

    def retrieve_dummy_words_list(self, expected_length):
        """
        A helper function which returns a random selection (of :param expected_length:) of the words in the vocabulary,
         specifically to be used for the dummy dataset
        """
        vocab_types = self.get_types()
        words = []
        while len(words) != expected_length:
            token = choice(vocab_types)
            if token not in [self.pad_word, self.bos_word, self.eos_word]:
                words.append(token)
        return words
