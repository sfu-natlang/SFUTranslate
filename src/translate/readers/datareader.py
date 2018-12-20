from abc import ABC, abstractmethod
from typing import Callable, Iterable
from sacrebleu import sentence_bleu

from translate.readers.constants import ReaderLevel, ReaderType
from translate.readers.vocabulary import Vocab
from translate.configs.loader import ConfigLoader
from translate.models.backend.utils import tensor2list


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
        self.source_vocabulary = Vocab(configs)
        self.target_vocabulary = Vocab(configs)

    @staticmethod
    def _sentensify(vocabulary: Vocab, ids: Iterable[int], merge_bpe_tokens: bool = False, input_is_tensor=False):
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
        return self._sentensify(self.target_vocabulary, ids, merge_bpe_tokens, input_is_tensor)

    def target_sentensify_all(self, ids_list: Iterable[Iterable[int]],
                              merge_bpe_tokens: bool = False, input_is_tensor=False):
        return [self._sentensify(self.target_vocabulary, ids, merge_bpe_tokens, input_is_tensor) for ids in ids_list]

    def compute_bleu(self, ref_ids_list: Iterable[Iterable[int]], hyp_ids_list: Iterable[Iterable[int]],
                     ref_is_tensor: bool=False, hyp_is_tensor: bool=False) -> float:
        assert len(ref_ids_list) == len(hyp_ids_list)
        refs = self.target_sentensify_all(ref_ids_list, input_is_tensor=ref_is_tensor)
        hyps = self.target_sentensify_all(hyp_ids_list, input_is_tensor=hyp_is_tensor)
        scores = [sentence_bleu(hyps[sid], refs[sid]) for sid in range(len(ref_ids_list))]
        return sum(scores) / len(scores)

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod
    def allocate(self):
        raise NotImplementedError

    @abstractmethod
    def deallocate(self):
        raise NotImplementedError

    def max_sentence_length(self):
        return self.max_valid_length
