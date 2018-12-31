"""
The wrapper classes over readers are supposed to be placed in this file. The wrappers could add extra information to 
 each instance while providing the same reader functionalities.    
"""
import numpy as np

from translate.logging.utils import logger
from translate.readers.constants import InstancePartType
from translate.readers.datareader import AbsDatasetReader

__author__ = "Hassan S. Shavarani"


class TransformerReaderWrapper(AbsDatasetReader):
    """
    The reader wrapper which add the transformer mask tensors (still in python list format in this class), to the input, 
     and target tensors passed to it. Please note that the wrapper considers the first two arguments omitted by the 
      passed reader as the input and target tensor values, and adds the mask tensors to the end of whatever number of 
       tensors passed to it. Please also note that this wrapper is meant to shadow all the functionalities of the actual 
        passed wrapper except that it adds the necessary transformer mask tensors to them. 
    """

    def __init__(self, data_reader: AbsDatasetReader):
        """
        :param data_reader: the reader which is able to read the dataset and provide input and target tensors to be 
         processed by this wrapper class 
        """
        self.data_provider = data_reader
        super(TransformerReaderWrapper, self).__init__(data_reader.configs, data_reader.reader_type, None, None)
        self.source_vocabulary = self.data_provider.source_vocabulary
        self.target_vocabulary = self.data_provider.target_vocabulary
        self.item_size = len(self.data_provider.instance_schema)
        if self.item_size > 2:
            raise NotImplementedError(
                "The wrapper has not been tested with data readers with more than 2 instance parts!")
        self.src_pad_idx = self.data_provider.source_vocabulary.get_pad_word_index()
        self.tgt_pad_idx = self.data_provider.target_vocabulary.get_pad_word_index()

    def get_sharable_data(self):
        return self.data_provider.get_sharable_data()

    def load_shared_reader_data(self, shared_data):
        if shared_data is not None and self.data_provider is not None:
            self.data_provider.load_shared_reader_data(shared_data)
        elif shared_data is not None:
            logger.warn("The Transformer Wrapper cannot load the shared data reader, check for potential errors!")

    def __len__(self):
        return len(self.data_provider)

    def deallocate(self):
        self.data_provider.deallocate()

    def allocate(self):
        self.data_provider.allocate()

    def __next__(self):
        n_item = next(self.data_provider)
        # TODO return ntokens = (self.trg != pad and bos).data.sum()
        if self.item_size == 1:
            src = np.array(n_item)
            src_mask = (src != self.src_pad_idx).astype('uint8').tolist()
            return n_item, src_mask
        else:
            src = np.array(n_item[0])
            tgt = np.array([self.target_vocabulary.get_begin_word_index()] + n_item[1])
            tgt_without_eos = tgt[:-1]
            tgt_without_bos = tgt[1:]
            src_mask = [(src != self.src_pad_idx).astype('uint8').tolist()]
            tgt_mask = (tgt_without_eos != self.tgt_pad_idx) & (
                np.triu(np.ones((tgt_without_eos.shape[-1], tgt_without_eos.shape[-1])), k=1).astype('uint8') == 0)
            tgt_mask = tgt_mask.astype('uint8').tolist()
            return [n_item[0], tgt_without_eos.tolist(), tgt_without_bos.tolist()] + \
                list(n_item[2:]) + [src_mask, tgt_mask]

    def __getitem__(self, idx):
        return next(self)

    def max_sentence_length(self):
        return self.data_provider.max_sentence_length()

    @property
    def instance_schema(self):
        scm = self.data_provider.instance_schema
        return [scm[0], scm[1], scm[1]] + list(scm[2:]) + [InstancePartType.TransformerSrcMask,
                                                           InstancePartType.TransformerTgtMask]
