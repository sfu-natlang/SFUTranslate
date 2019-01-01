"""
The implementation of the parallel dataset reader from a directory containing the preprocessed train/dev/test data.
 The directory must contain two files named the same ending into language name identifiers the following line shows the 
  list of files for an example valid dataset directory.
  - train.normalized.en
  - train.normalized.fr
  - test.normalized.en
  - test.normalized.fr
  - dev.normalized.en
  - dev.normalized.fr
The reader can be configured via setting the following values in the config file to the desired values.
##################################################
reader:
    dataset:
        type: parallel
        buffer_size: 10000
        max_length: 128
        source_lang: fr
        target_lang: en
        working_dir: /path/to/dataset
        train_file_name: train.normalized
        test_file_name: test.normalized
        dev_file_name: dev.normalized
##################################################
"""
from random import shuffle
from typing import Callable, Dict
from pathlib import Path
from collections import Counter

from translate.configs.loader import ConfigLoader
from translate.readers.constants import ReaderType, InstancePartType
from translate.readers.datareader import AbsDatasetReader
from translate.configs.utils import get_dataset_file
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


class ParallelDataReader(AbsDatasetReader):
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
        super(ParallelDataReader, self).__init__(configs, reader_type, iter_log_handler, shared_reader_data)
        src_lang = configs.get("reader.dataset.source_lang", must_exist=True)
        tgt_lang = configs.get("reader.dataset.target_lang", must_exist=True)
        w_dir = configs.get("reader.dataset.working_dir", must_exist=True)
        if reader_type == ReaderType.TRAIN:
            file_name = configs.get("reader.dataset.train_file_name", must_exist=True)
        elif reader_type == ReaderType.TEST:
            file_name = configs.get("reader.dataset.test_file_name", must_exist=True)
        elif reader_type == ReaderType.DEV:
            file_name = configs.get("reader.dataset.dev_file_name", must_exist=True)
        else:
            raise NotImplementedError
        self.source_file = get_dataset_file(w_dir, file_name, src_lang)
        self.target_file = get_dataset_file(w_dir, file_name, tgt_lang)
        if reader_type == ReaderType.TRAIN:
            self.source_vocabulary.set_types(self.load_vocab_counts(
                self.source_file, get_dataset_file(w_dir, "vocab_counts", src_lang)))
            self.target_vocabulary.set_types(self.load_vocab_counts(
                self.target_file, get_dataset_file(w_dir, "vocab_counts", tgt_lang)))
        self.files_opened = False
        self.source = None
        self.target = None
        self._buffer = None
        source_lines_count = sum((1 for _ in self.source_file.open()))
        target_lines_count = sum((1 for _ in self.target_file.open()))
        assert source_lines_count == target_lines_count
        self.lines_count = source_lines_count

    @staticmethod
    def load_vocab_counts(data_file: Path, vocab_counts_file: Path, min_count: int=1):
        """
        The method to take a :param vocab_counts_file: to load the vocabulary from (the words above :param min_count: 
         will get loaded from the file). The method will create the file if it does not exist by going through the 
          actual resource :param data_file: to collect the vocabulary.
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
            with data_file.open() as dtf:
                for line in dtf:
                    lines_count += 1
                    for word in line.strip().split():
                        vocab_counts[word] += 1
            with vocab_counts_file.open(mode='w') as vcf:
                for word, count in vocab_counts.most_common():
                    vcf.write("{} {}\n".format(word, count))
        result = []
        with vocab_counts_file.open() as existing_vocab_file:
            for line in existing_vocab_file:
                line_parts = line.split()
                word = line_parts[0]
                count = int(line_parts[1])
                if count > min_count:
                    result.append(word)
        return result

    def get_sharable_data(self):
        return {"source_types": self.source_vocabulary.get_types(), "target_types": self.target_vocabulary.get_types()}

    def deallocate(self):
        self.files_opened = False
        self.source = None
        self.target = None
        self._buffer = None

    def __len__(self):
        return self.lines_count

    @property
    def instance_schema(self):
        return InstancePartType.ListId, InstancePartType.ListId

    def load_shared_reader_data(self, shared_data):
        if self.reader_type != ReaderType.TRAIN and shared_data is None:
            raise ValueError("Only trainer instance is allowed to create the vocabulary from the sentences sentences!")
        if shared_data is not None:
            self.source_vocabulary.set_types(shared_data["source_types"])
            self.target_vocabulary.set_types(shared_data["target_types"])

    def __getitem__(self, idx):
        # TODO make sure the same index is not referred to again
        return next(self)

    def allocate(self):
        self.files_opened = True
        self.source = self.source_file.open()
        self.target = self.target_file.open()
        self._buffer = []

    def max_sentence_length(self):
        return self._max_valid_length

    def __next__(self):
        if not len(self._buffer):
            if not self.files_opened:
                logger.error("You might need to call \"allocate()\" first!")
                raise StopIteration
            for src, tgt in zip(self.source, self.target):
                src_ids = [self.source_vocabulary[x] for x in src.strip().split()]
                tgt_ids = [self.target_vocabulary[x] for x in tgt.strip().split()]
                src_len = len(src_ids)
                tgt_len = len(tgt_ids)
                if src_len > self._max_valid_length or not src_len or tgt_len > self._max_valid_length or not tgt_len:
                    continue
                self._buffer.append((src_ids, tgt_ids, src_len+tgt_len))
                if len(self._buffer) == self._instance_buffer_size:
                    break
                shuffle(self._buffer)
                self._buffer = sorted(self._buffer, key=lambda element: element[2], reverse=True)
        if not len(self._buffer):
            raise StopIteration
        src, tgt, _ = self._buffer.pop(0)
        return src, tgt
