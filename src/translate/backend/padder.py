"""
The class in charge of padding , batching, and post-processing of the created instances in the dataset reader
"""
from typing import Union, List, Tuple

from translate.readers.constants import InstancePartType
from translate.readers.datareader import AbsDatasetReader
from translate.backend.utils import device, backend, DataLoader, zeros_tensor

__author__ = "Hassan S. Shavarani"


def _pad_transform_id_list(id_list: Union[List, backend.Tensor], max_length: int, pad_token_id: int) -> backend.Tensor:
    """
    Receives the word-indices [integer/long type] and converts them into a backend tensor
    """
    assert len(id_list) > 0
    assert type(id_list[0]) is not list
    if type(id_list) == list:
        id_list.extend([pad_token_id] * (max_length - len(id_list)))
        result = backend.LongTensor(id_list, device=device)
        if backend.cuda.is_available():
            return result.cuda()
        else:
            return result
    else:
        result = id_list.view(-1)
        pad_size = max_length - result.size(0)
        if pad_size > 0:
            pad_vec = backend.ones(pad_size).long().to(device) * pad_token_id
            result = backend.cat([result, pad_vec], dim=0)
        if backend.cuda.is_available():
            return result.cuda()
        else:
            return result


def _pad_transform_list_list_id(id_list: List[List[int]], max_length: int, pad_token_id: int,
                                row_wise_padding: bool = False) -> backend.Tensor:
    """
    Receives a list of list of word-indices [integer/long type] and converts them into a backend tensor
    """
    result = backend.stack([_pad_transform_id_list(x, max_length, pad_token_id) for x in id_list], dim=0)
    if len(id_list) < max_length and row_wise_padding:
        return backend.cat([result, zeros_tensor(1, max_length - len(id_list), max_length).long().squeeze(0)], dim=0)
    else:
        return result


def _pad_transform_embedding_matrix(embedding_matrix: backend.Tensor, max_length: int) -> backend.Tensor:
    """
    :param embedding_matrix: A matrix of size [SentenceLength + 1, EmbeddingSize]
     The +1 part is because the embedding is always ended in embedding vector of pad_word
    :param max_length: the max length to which the matrix is supposed to be padded to
    """
    assert embedding_matrix.dim() == 2
    result = embedding_matrix[:-1]
    pad_size = max_length - result.size(0)
    if pad_size > 0:
        result = backend.cat([result, embedding_matrix[-1].repeat(pad_size, 1)])
    if backend.cuda.is_available():
        return result.cuda()
    else:
        return result


def get_padding_batch_loader(dataset_instance: AbsDatasetReader, batch_size: int) -> DataLoader:
    """
    :returns a DataLoader which takes single instances from :param dataset_instance:, batches them into batches of
     size :param batch_size:, pads them and returns the batches in the format of an iterator
    """
    return DataLoader(dataset_instance, batch_size=batch_size,
                      collate_fn=PadCollate(pad_index_e=dataset_instance.target_vocabulary.get_pad_word_index(),
                                            pad_index_f=dataset_instance.source_vocabulary.get_pad_word_index(),
                                            instance_schema=dataset_instance.instance_schema))


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in a batch of sequences
    """

    def __init__(self, pad_index_f: int, pad_index_e: int, instance_schema: Tuple):
        """
        receives the padding indices which will be used to pad the tensors
        """
        self.pad_index_e = pad_index_e
        self.pad_index_f = pad_index_f
        self.instance_schema = instance_schema

    @staticmethod
    def get_item_length(id_list: Union[backend.Tensor, List], schema_type: InstancePartType) -> int:
        """
        given a list or a tensor, the function will detect the batch size in it and will return it
        """
        if schema_type == InstancePartType.Tensor and id_list.size(0) == 1:
            return id_list.view(-1).size(0)
        elif schema_type == InstancePartType.Tensor:
            raise NotImplementedError
        elif schema_type == InstancePartType.ListId:
            return len(id_list)
        elif schema_type == InstancePartType.TransformerSrcMask or \
                        schema_type == InstancePartType.TransformerTgtMask:
            return len(id_list[0])
        else:
            raise NotImplementedError("Unknown schema type {}".format(schema_type))

    def pad_collate(self, batch) -> Tuple:
        """
        the function to perform the padding + batching + conversion of final resulting batch to a tensor
        :param batch: a batch of Tuples of inputs
         every single input Tuple can contain a number of list (tensors) of ids
        """
        result = None
        for ind, schema_type in enumerate(self.instance_schema):
            max_len = max(map(lambda x: self.get_item_length(x[ind], schema_type), batch))
            pad_index = 0
            if ind == 0:
                pad_index = self.pad_index_f
            elif ind == 1:
                pad_index = self.pad_index_e
            if schema_type == InstancePartType.ListId:
                res = backend.stack([x for x in map(lambda p: (
                    _pad_transform_id_list(p[ind], max_len, pad_index)), batch)], dim=0)
            elif schema_type == InstancePartType.TransformerSrcMask or \
                            schema_type == InstancePartType.TransformerTgtMask:
                res = backend.stack([x for x in map(lambda p: (_pad_transform_list_list_id(
                    p[ind], max_len, pad_index, schema_type == InstancePartType.TransformerTgtMask)), batch)], dim=0)
            else:
                raise NotImplementedError
            if result is None:
                result = res,
            else:
                result = *result, res
        return result

    def pad_collate_deprecated(self, batch) -> Tuple:
        """
        the function to perform the padding + batching + conversion of final resulting batch to a tensor
        :param batch: a batch of Tuples of inputs
         every single input Tuple can contain a number of list (tensors) of ids
        """
        # find longest sequence
        batch_elements_size = len(batch[0])
        max_len_f = max(map(lambda x: self.get_item_length(x[0]), batch))
        # pad according to max_len
        if batch_elements_size == 1:
            batch = map(lambda p: (_pad_transform_id_list(p[0], max_len_f, self.pad_index_f)), batch)
            res_f = backend.stack([x for x in batch], dim=0)
            return res_f,
        elif batch_elements_size == 2:
            max_len_e = max(map(lambda x: self.get_item_length(x[1]), batch))
            batch = [item for item in map(lambda p: (_pad_transform_id_list(p[0], max_len_f, self.pad_index_f),
                                                     _pad_transform_id_list(p[1], max_len_e, self.pad_index_e)), batch)]
            res_f = backend.stack([x for x in map(lambda x: x[0], batch)], dim=0)
            res_e = backend.stack([x for x in map(lambda x: x[1], batch)], dim=0)
            return res_f, res_e
        elif batch_elements_size == 3:
            max_len_e = max(map(lambda x: self.get_item_length(x[1]), batch))
            max_len_g = max(map(lambda x: self.get_item_length(x[2]), batch))
            batch = [item for item in map(lambda p: (_pad_transform_id_list(p[0], max_len_f, self.pad_index_f),
                                                     _pad_transform_id_list(p[1], max_len_e, self.pad_index_e),
                                                     _pad_transform_embedding_matrix(p[2], max_len_g)), batch)]
            res_f = backend.stack([x for x in map(lambda x: x[0], batch)], dim=0)
            res_e = backend.stack([x for x in map(lambda x: x[1], batch)], dim=0)
            res_g = backend.stack([x for x in map(lambda x: x[2], batch)], dim=0)
            return res_f, res_e, res_g

    def __call__(self, batch):
        return self.pad_collate(batch)
