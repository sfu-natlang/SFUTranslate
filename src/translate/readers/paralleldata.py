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
from collections import deque
from enum import Enum
from random import shuffle
from typing import Dict

from translate.configs.loader import ConfigLoader
from translate.readers.constants import ReaderType, InstancePartType, ReaderLevel
from translate.readers.datareader import AbsDatasetReader, DatasetStats
from translate.configs.utils import get_dataset_file
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


class ParallelSide(Enum):
    SOURCE = 0
    TARGET = 1


class ParallelDataReader(AbsDatasetReader):
    def __init__(self, configs: ConfigLoader, reader_type: ReaderType, shared_reader_data: Dict = None):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param reader_type: an instance of ReaderType enum stating the type of the dataste (e.g. Train, Test, Dev) 
        :param shared_reader_data: the data shared from another reader to this reader instance
        """
        super(ParallelDataReader, self).__init__(configs, reader_type, shared_reader_data)
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
        # word granularity can be used in the dataset reader to prepare the data in a specific format
        self._src_word_granularity = ReaderLevel.get_granularity(
            configs.get("reader.dataset.granularity.src", default_value="WORD"))
        self._tgt_word_granularity = ReaderLevel.get_granularity(
            configs.get("reader.dataset.granularity.tgt", default_value="WORD"))
        self.source_file = get_dataset_file(w_dir, file_name, src_lang)
        self.target_file = get_dataset_file(w_dir, file_name, tgt_lang)
        if reader_type == ReaderType.TRAIN:
            logger.info("Source input granularity: {}".format(self._src_word_granularity))
            logger.info("Target input granularity: {}".format(self._tgt_word_granularity))
            logger.info("Reader maximum valid input length: {}".format(self._max_valid_length))
            if self._src_word_granularity == ReaderLevel.BPE:
                src_merge_size = configs.get("reader.vocab.bpe_merge_size.src", must_exist=True)
                self._source_bpe_model = self.retrieve_bpe_model(self.source_file, src_merge_size,
                                                                 self.source_vocabulary.bpe_separator)
            if self._tgt_word_granularity == ReaderLevel.BPE:
                tgt_merge_size = configs.get("reader.vocab.bpe_merge_size.tgt", must_exist=True)

                self._target_bpe_model = self.retrieve_bpe_model(self.target_file, tgt_merge_size,
                                                                 self.target_vocabulary.bpe_separator)
            min_count_src = configs.get("reader.vocab.min_count.src", 1)
            min_count_tgt = configs.get("reader.vocab.min_count.tgt", 1)
            self.source_vocabulary.set_types(self.load_vocab_counts(self.get_resource_lines(
                ParallelSide.SOURCE), get_dataset_file(w_dir, "vocab_counts_{}".format(
                    self._src_word_granularity.name.lower()), src_lang), min_count=min_count_src))
            self.target_vocabulary.set_types(self.load_vocab_counts(self.get_resource_lines(
                ParallelSide.TARGET), get_dataset_file(w_dir, "vocab_counts_{}".format(
                    self._tgt_word_granularity.name.lower()), tgt_lang), min_count=min_count_tgt))
            logger.info("Source vocabulary loaded with size |F|= %d (min word count=%d)" % (
                len(self.source_vocabulary), min_count_src))
            logger.info("Target vocabulary loaded with size |E|= %d (min word count=%d)" % (
                len(self.target_vocabulary), min_count_tgt))
        self.files_opened = False
        self.source = None
        self.target = None
        self._buffer = None
        self._buffer_token_size_src = 0
        self._buffer_token_size_tgt = 0
        self._temporary_buffer = deque([])
        source_lines_count = sum((1 for line in self.source_file.open() if len(line.strip())))
        target_lines_count = sum((1 for line in self.target_file.open() if len(line.strip())))
        assert source_lines_count == target_lines_count
        self.lines_count = source_lines_count
        self.source_stats = DatasetStats()
        self.target_stats = DatasetStats()
        logger.info("Parallel data lines in {}: {}".format(self.reader_type.name, self.lines_count))

    def get_resource_lines(self, side: ParallelSide):
        """
        The method which goes through the source/target resource(s) and pre-processes them considering
         the granularity and returns the pre-processed lines
        """
        if side == ParallelSide.SOURCE:
            resource_file = self.source_file
            bpe_model = self._source_bpe_model
            space_word = self.source_vocabulary.space_word
            g_level = self._src_word_granularity
        else:
            resource_file = self.target_file
            bpe_model = self._target_bpe_model
            space_word = self.target_vocabulary.space_word
            g_level = self._tgt_word_granularity
        for line in resource_file.open():
            if g_level == ReaderLevel.BPE:
                yield bpe_model.segment(line)
            elif g_level == ReaderLevel.CHAR:
                yield " ".join([x for x in line.strip().replace(" ", "☺")]).replace("☺", space_word)
            else:
                yield line

    def get_sharable_data(self):
        return {"source_types": self.source_vocabulary.get_types(), "target_types": self.target_vocabulary.get_types(),
                "source_bpe_model": self._source_bpe_model, "target_bpe_model": self._target_bpe_model}

    def deallocate(self):
        self.files_opened = False
        self.source = None
        self.target = None
        self._buffer = None
        # The following lines will run only once, spitting out logs about the dataset that was just processed.
        #  The log lines show the minimum, average and maximum line size of source and target sentences after the moment
        #   that the end of sentence got added to them.
        if self.source_stats is not None:
            logger.info("{}.SOURCE stats: [MinLS: {:.2f}, AvgLS: {:.2f}, MaxLS:{:.2f}]".format(
                self.reader_type.name, self.source_stats.min_size, self.source_stats.avg_size,
                self.source_stats.max_size))
            self.source_stats = None
        if self.target_stats is not None:
            logger.info("{}.TARGET stats: [MinLS: {:.2f}, AvgLS: {:.2f}, MaxLS:{:.2f}]".format(
                self.reader_type.name, self.target_stats.min_size, self.target_stats.avg_size,
                self.target_stats.max_size))
            self.target_stats = None

    def __len__(self):
        return self.lines_count

    @property
    def instance_schema(self):
        return InstancePartType.ListId, InstancePartType.ListId

    @property
    def bfp(self):
        """
        :return: the filling percentage of buffer out of 100%
        """
        return "{:.1f}".format(len(self._temporary_buffer) * 100.0 / self._instance_buffer_size)

    def load_shared_reader_data(self, shared_data):
        if self.reader_type != ReaderType.TRAIN and shared_data is None:
            raise ValueError("Only trainer instance is allowed to create the vocabulary from the sentences sentences!")
        if shared_data is not None:
            self.source_vocabulary.set_types(shared_data["source_types"])
            self.target_vocabulary.set_types(shared_data["target_types"])
            self._source_bpe_model = shared_data["source_bpe_model"]
            self._target_bpe_model = shared_data["target_bpe_model"]

    def __getitem__(self, idx):
        # If you are testing, make sure the same index is not referred to again
        return next(self)

    def allocate(self):
        self.files_opened = True
        self.source = self.get_resource_lines(ParallelSide.SOURCE)
        self.target = self.get_resource_lines(ParallelSide.TARGET)
        self._buffer = []

    def max_sentence_length(self):
        return self._max_valid_length

    def __next__(self):
        if not len(self._buffer):
            if not self.files_opened:
                logger.error("You might need to call \"allocate()\" first!")
                raise StopIteration
            temporary_buffer_token_size_src = 0
            temporary_buffer_token_size_tgt = 0
            for src, tgt in zip(self.source, self.target):
                src_ids = [self.source_vocabulary[x] for x in src.strip().split()]
                tgt_ids = [self.target_vocabulary[x] for x in tgt.strip().split()]
                src_ids += [self.source_vocabulary.get_end_word_index()]
                tgt_ids += [self.target_vocabulary.get_end_word_index()]
                src_len = len(src_ids)
                tgt_len = len(tgt_ids)
                if src_len > self._max_valid_length or not src_len or tgt_len > self._max_valid_length or not tgt_len:
                    continue
                if self._iter_log_handler is not None:
                    self._iter_log_handler("{}: Filling Reader Buffer [rfp: {}%]".format(
                        self.reader_type.name, self.bfp))
                if self.source_stats is not None:
                    self.source_stats.update(src_len)
                if self.target_stats is not None:
                    self.target_stats.update(tgt_len)
                self._temporary_buffer.append((src_ids, tgt_ids, src_len + tgt_len))
                temporary_buffer_token_size_src += src_len
                temporary_buffer_token_size_tgt += tgt_len
                if len(self._temporary_buffer) == self._instance_buffer_size:
                    break
            self._buffer.extend(list(self._temporary_buffer))
            self._buffer_token_size_src += temporary_buffer_token_size_src
            self._buffer_token_size_tgt += temporary_buffer_token_size_tgt
            shuffle(self._buffer)
            self._buffer = sorted(self._buffer, key=lambda element: element[2], reverse=True)
            del self._temporary_buffer
            self._temporary_buffer = deque([])
        if not len(self._buffer):
            raise StopIteration
        src, tgt, _ = self._buffer.pop(0)
        return src, tgt
