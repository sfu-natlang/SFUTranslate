from translate.models.backend.utils import device, backend, DataLoader


def pad_transform_id_list(id_list, max_length, pad_token_id):
    """
    Receives the word-indexs [integer/long type] and converts them into a backend tensor
    """
    # TODO make sure the output is always of the same size and input is same formatted
    assert type(id_list) == list or type(id_list) == backend.Tensor
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


def pad_transform_embedding_matrix(embedding_matrix, max_length):
    """
    :param embedding_matrix: A matrix of size [SentenceLength + 1, EmbeddingSize]
     The +1 part is because the embedding is always ended in embedding vector of pad_word
    :param max_length: the max length to which the matrix is supposed to be padded to
    :return:
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


def get_padding_batch_loader(dataset_instance, batch_size):
    return DataLoader(dataset_instance, batch_size=batch_size,
                      collate_fn=PadCollate(pad_index_e=dataset_instance.target_vocabulary.get_pad_word_index(),
                                            pad_index_f=dataset_instance.source_vocabulary.get_pad_word_index()))


class PadCollate:
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, pad_index_f, pad_index_e):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.pad_index_e = pad_index_e
        self.pad_index_f = pad_index_f

    @staticmethod
    def get_item_length(id_list):
        if type(id_list) == backend.Tensor and id_list.size(0) == 1:
            return id_list.view(-1).size(0)
        else:
            return len(id_list)

    def pad_collate(self, batch):
        """
        args:
            batch - list of (input tensor, label)
        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        batch_elements_size = len(batch[0])
        max_len_f = max(map(lambda x: self.get_item_length(x[0]), batch))
        # pad according to max_len
        if batch_elements_size == 1:
            batch = map(lambda p: (pad_transform_id_list(p[0], max_len_f, self.pad_index_f)), batch)
            res_f = backend.stack([x for x in map(lambda x: x[0], batch)], dim=0)
            return res_f
        elif batch_elements_size == 2:
            max_len_e = max(map(lambda x: self.get_item_length(x[1]), batch))
            batch = [item for item in map(lambda p: (pad_transform_id_list(p[0], max_len_f, self.pad_index_f),
                                                     pad_transform_id_list(p[1], max_len_e, self.pad_index_e)), batch)]
            res_f = backend.stack([x for x in map(lambda x: x[0], batch)], dim=0)
            res_e = backend.stack([x for x in map(lambda x: x[1], batch)], dim=0)
            return res_f, res_e
        elif batch_elements_size == 3:
            max_len_e = max(map(lambda x: self.get_item_length(x[1]), batch))
            max_len_g = max(map(lambda x: self.get_item_length(x[2]), batch))
            batch = [item for item in map(lambda p: (pad_transform_id_list(p[0], max_len_f, self.pad_index_f),
                                                     pad_transform_id_list(p[1], max_len_e, self.pad_index_e),
                                                     pad_transform_embedding_matrix(p[2], max_len_g)), batch)]
            res_f = backend.stack([x for x in map(lambda x: x[0], batch)], dim=0)
            res_e = backend.stack([x for x in map(lambda x: x[1], batch)], dim=0)
            res_g = backend.stack([x for x in map(lambda x: x[2], batch)], dim=0)
            return res_f, res_e, res_g

    def __call__(self, batch):
        return self.pad_collate(batch)