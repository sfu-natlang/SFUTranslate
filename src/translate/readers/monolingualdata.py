"""
The implementation of the monolingual dataset reader from a directory containing the preprocessed train/dev/test data.
 The directory must contain one file for each of train/test/dev cases ending into the language name identifier.
  the following line shows the list of files for an example valid dataset directory.
  - train.normalized.en
  - test.normalized.en
  - dev.normalized.en
The reader can be configured via setting the following values in the config file to the desired values.
NOTE: this reader class is a special case of Parallel data reader in which the target side is null. Due to this reason
 and also due to simplicity of configurations the same config tags as parallel data reader are used in the "src" part,
  to configure the instances of this class
##################################################
reader:
    dataset:
        type: mono
        buffer_size: 10000
        max_length: 128
        source_lang: en
        working_dir: /path/to/dataset
        train_file_name: train.normalized
        test_file_name: test.normalized
        dev_file_name: dev.normalized
##################################################
"""
from random import shuffle
from typing import Dict

from translate.configs.loader import ConfigLoader
from translate.readers.constants import ReaderType, InstancePartType, ReaderLevel
from translate.readers.datareader import AbsDatasetReader, DatasetStats
from translate.configs.utils import get_dataset_file
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


class MonolingualDataReader(AbsDatasetReader):
    def __init__(self, configs: ConfigLoader, reader_type: ReaderType, shared_reader_data: Dict = None):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param reader_type: an instance of ReaderType enum stating the type of the dataste (e.g. Train, Test, Dev) 
        :param shared_reader_data: the data shared from another reader to this reader instance
        """
        super(MonolingualDataReader, self).__init__(configs, reader_type, shared_reader_data)
        src_lang = configs.get("reader.dataset.source_lang", must_exist=True)
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
        self._word_granularity = ReaderLevel.get_granularity(
            configs.get("reader.dataset.granularity.src", default_value="WORD"))
        self.data_file = get_dataset_file(w_dir, file_name, src_lang)
        if reader_type == ReaderType.TRAIN:
            logger.info("Reader input granularity: {}".format(self._word_granularity))
            logger.info("Reader maximum valid input length: {}".format(self._max_valid_length))
            if self._word_granularity == ReaderLevel.BPE:
                src_merge_size = configs.get("reader.vocab.bpe_merge_size.src", must_exist=True)
                self._source_bpe_model = self.retrieve_bpe_model(self.data_file, src_merge_size,
                                                                 self.source_vocabulary.bpe_separator)
            min_count_src = configs.get("reader.vocab.min_count.src", 1)
            self.source_vocabulary.set_types(self.load_vocab_counts(self.get_resource_lines(), get_dataset_file(
                w_dir, "vocab_counts_{}".format(
                    self._word_granularity.name.lower()), src_lang), min_count=min_count_src))
            self.target_vocabulary.set_types(self.source_vocabulary.get_types())
            logger.info("Vocabulary loaded with size |V| = %d (min word count=%d)" % (
                len(self.source_vocabulary), min_count_src))
        self.files_opened = False
        self.data_stream = None
        self._buffer = None
        self.lines_count = sum((1 for _ in self.data_file.open()))
        self.data_stats = DatasetStats()

    def get_resource_lines(self):
        """
        The method which goes through the source resource(s) and pre-processes them considering the granularity and
         returns the pre-processed lines
        """
        for line in self.data_file.open():
            if self._word_granularity == ReaderLevel.BPE:
                yield self._source_bpe_model.segment(line)
            elif self._word_granularity == ReaderLevel.CHAR:
                yield " ".join([x for x in line.strip().replace(" ", self.source_vocabulary.space_word)])
            else:
                yield line

    def get_sharable_data(self):
        return {"source_types": self.source_vocabulary.get_types(), "source_bpe_model": self._source_bpe_model}

    def deallocate(self):
        self.files_opened = False
        self.data_stream = None
        self._buffer = None
        # The following lines will run only once, spitting out logs about the dataset that was just processed.
        #  The log lines show the minimum, average and maximum line size of source sentences after the moment
        #   that the end of sentence got added to them.
        if self.data_stats is not None:
            logger.info("{}.SOURCE stats: [MinLS: {:.2f}, AvgLS: {:.2f}, MaxLS:{:.2f}]".format(
                self.reader_type.name, self.data_stats.min_size, self.data_stats.avg_size,
                self.data_stats.max_size))
            self.data_stats = None

    def __len__(self):
        return self.lines_count

    @property
    def instance_schema(self):
        return InstancePartType.ListId,

    @property
    def bfp(self):
        """
        :return: the filling percentage of buffer out of 100%
        """
        return "{:.1f}".format(len(self._buffer) * 100.0 / self._instance_buffer_size)

    def load_shared_reader_data(self, shared_data):
        if self.reader_type != ReaderType.TRAIN and shared_data is None:
            raise ValueError("Only trainer instance is allowed to create the vocabulary from the sentences sentences!")
        if shared_data is not None:
            self.source_vocabulary.set_types(shared_data["source_types"])
            self.target_vocabulary.set_types(self.source_vocabulary.get_types())
            self._source_bpe_model = shared_data["source_bpe_model"]

    def __getitem__(self, idx):
        # If you are testing, make sure the same index is not referred to again
        return next(self)

    def allocate(self):
        self.files_opened = True
        self.data_stream = self.get_resource_lines()
        self._buffer = []

    def max_sentence_length(self):
        return self._max_valid_length

    def __next__(self):
        if not len(self._buffer):
            if not self.files_opened:
                logger.error("You might need to call \"allocate()\" first!")
                raise StopIteration
            for src in self.data_stream:
                src_ids = [self.source_vocabulary[x] for x in src.strip().split()]
                src_ids += [self.source_vocabulary.get_end_word_index()]
                src_len = len(src_ids)
                if src_len > self._max_valid_length or not src_len:
                    continue
                if self._iter_log_handler is not None:
                    self._iter_log_handler("{}: Filling Reader Buffer [rfp: {}%]".format(
                        self.reader_type.name, self.bfp))
                if self.data_stats is not None:
                    self.data_stats.update(src_len)
                self._buffer.append((src_ids, src_len))
                if len(self._buffer) == self._instance_buffer_size:
                    break
                shuffle(self._buffer)
                self._buffer = sorted(self._buffer, key=lambda element: element[1], reverse=True)
        if not len(self._buffer):
            raise StopIteration
        src, _ = self._buffer.pop(0)
        return src,
