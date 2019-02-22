"""
The utility class helping to collect results and keep track of them, and export the decoded tensors into final lists of 
 word ids
"""
from translate.backend.utils import backend, device

__author__ = "Hassan S. Shavarani"


class _DecodingSentence:
    def __init__(self, eos_id, pad_id):
        self.word_ids = []
        self.eos_id = eos_id
        self.pad_id = pad_id
        self._eos_reached = False
        self.sentence_probability = 0

    def clone(self):
        cln = _DecodingSentence(self.eos_id, self.pad_id)
        cln.word_ids = [wid for wid in self.word_ids]
        cln._eos_reached = self._eos_reached
        cln.sentence_probability = self.sentence_probability
        return cln

    def append(self, word_ids: backend.Tensor, word_id_log_probabilities: backend.Tensor):
        """
        :param word_ids: 1-D Tensor of size [beam_size=1] containing word ids
        :param word_id_log_probabilities: 1-D Tensor of size [beam_size=1] containing word ids
        """
        if word_ids.size(0) > 1:
            raise ValueError("Beam sizes bigger than 1 are not supported")
        word = word_ids.item()
        if word == self.eos_id:
            self._eos_reached = True
        self.word_ids.append(word)
        self.sentence_probability += word_id_log_probabilities

    @property
    def log_probability(self):
        return self.sentence_probability.item() / len(self.word_ids) if len(self.word_ids) else 0.0

    @property
    def eos_reached(self):
        return self._eos_reached

    @property
    def ids(self):
        if self.eos_id in self.word_ids:
            return [word for word in self.word_ids[:self.word_ids.index(self.eos_id)] if word != self.pad_id]
        else:
            return [word for word in self.word_ids if word != self.pad_id]

    def __del__(self):
        del self.word_ids


class DecodingResult:
    def __init__(self, batch_size, pad_id, eos_id):
        """
        :param batch_size: the expected batch size of the tensors passed to be kept track of
        :param pad_id: the target side pad id
        :param eos_id: the target size end of sentence id
        """
        self.batch_sentences = [_DecodingSentence(eos_id, pad_id) for _ in range(batch_size)]
        self.batch_size = batch_size

    def append(self, batch_id_probabilities: backend.Tensor, batch_ids: backend.Tensor):
        """
        :param batch_ids: 2-D Tensor of size [batch_size, beam_size]
        :param batch_id_probabilities: 2-D Tensor of size [batch_size, beam_size]
        :return:
        """
        for index in range(batch_ids.size(0)):
            self.batch_sentences[index].append(batch_ids[index], batch_id_probabilities[index])
        return batch_ids[:, 0].view(-1).to(device).detach()

    @property
    def decoding_completed(self):
        return self.batch_size == sum((sent.eos_reached for sent in self.batch_sentences))

    @property
    def ids(self):
        return [sent.ids for sent in self.batch_sentences]

    @property
    def log_probability(self):
        return sum((sent.log_probability for sent in self.batch_sentences))
