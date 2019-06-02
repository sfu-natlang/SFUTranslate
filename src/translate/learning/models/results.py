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
        if word_ids.dim() > 0 and word_ids.size(0) > 1:
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


class GreedyDecodingResult:
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


class BeamDecodingResult:
    def __init__(self, batch_size, beam_size, pad_id, eos_id):
        """
        :param batch_size: the expected batch size of the tensors passed to be kept track of
        :param beam_size: the expected beam_size to keep track of the result
        :param pad_id: the target side pad id
        :param eos_id: the target size end of sentence id
        """
        self.batch_sentences = [[_DecodingSentence(eos_id, pad_id) for _ in range(beam_size)]
                                for _ in range(batch_size)]
        self.batch_size = batch_size
        self.beam_size = beam_size

    def append(self, batch_id_probabilities: backend.Tensor, batch_ids: backend.Tensor):
        """
        :param batch_ids: 2-D Tensor of size [batch_size * beam_size, beam_size]
        :param batch_id_probabilities: 2-D Tensor of size [batch_size * beam_size, beam_size]
        :return: 1-D Tensor of size [batch_size * beam_size] in which each 'beam_size' number of elements together are 
            related to one sentence, a 1-D array of size `batch_size * beam_size` stating which previous state should be 
             kept for each element
        """
        result = []
        selection_bucket = []
        for index in range(self.batch_size):
            initial = index * self.beam_size
            end = (index + 1) * self.beam_size
            res, bucket_ids = self._beam_append(index, batch_ids[initial:end], batch_id_probabilities[initial:end])
            result.append(res)
            for sid in bucket_ids:
                selection_bucket.append(initial + sid)
        return backend.cat(result), selection_bucket

    def _beam_append(self, batch_index: int, beam_ids: backend.Tensor, beam_id_probabilities: backend.Tensor):
        """
        The function in charge of taking the results of the beam search for one sentence and prune the irrelevant ones
        :param batch_index: the sentence index in the batch
        :param beam_ids: 2-D Tensor of size [beam_size, beam_size]
        :param beam_id_probabilities: 2-D Tensor of size [beam_size, beam_size]
        :return: 1-D Tensor of size [beam_size] which shows the `beam_size` best number of ids produced by the model, 
         a 1-D array of size `beam_size` stating which previous state should be kept for each beam
        """
        sentence_beam = self.batch_sentences[batch_index]
        scores = []
        for i in range(self.beam_size):
            ith_beam_prob = sentence_beam[i].log_probability  # one instance of _DecodingSentence
            ith_beam_log_probabilities = beam_id_probabilities[i]  # 1-D Tensor of size [beam_size]
            for j in range(self.beam_size):
                scores.append((i, j, ith_beam_prob + ith_beam_log_probabilities[j].item()))
        selection = sorted(scores, key=lambda x: x[2], reverse=True)[:self.beam_size]
        new_beam = []
        resulting_ids = []
        selection_buckets = []
        for i, j, _ in selection:
            new_beam_element = sentence_beam[i].clone()
            new_beam_element.append(beam_ids[i][j], beam_id_probabilities[i][j])
            resulting_ids.append(beam_ids[i][j].unsqueeze(-1).to(device).detach())
            new_beam.append(new_beam_element)
            selection_buckets.append(i)
        del self.batch_sentences[batch_index][:]
        self.batch_sentences[batch_index] = new_beam
        return backend.cat(resulting_ids), selection_buckets

    @property
    def decoding_completed(self):
        return self.batch_size * self.beam_size == sum(
            (sent.eos_reached for beam_sent in self.batch_sentences for sent in beam_sent))

    @property
    def ids(self):
        result = []
        for beam_sent in self.batch_sentences:
            best_sent = None
            best_score = float("-inf")
            for sent in beam_sent:
                avg_token_prob = sent.log_probability / len(sent.ids)
                if avg_token_prob > best_score:
                    best_score = avg_token_prob
                    best_sent = sent
            if best_sent is not None:  # should be always true
                result.append(best_sent.ids)
            else:
                result.append([])
        return result

    @property
    def log_probability(self):
        result = []
        for beam_sent in self.batch_sentences:
            best_score = float("-inf")
            best_scoring_len = 0.0
            best_score_found = False
            for sent in beam_sent:
                avg_token_prob = sent.log_probability / len(sent.ids)
                if avg_token_prob > best_score:
                    best_score = avg_token_prob
                    best_score_found = True
                    best_scoring_len = len(sent.ids)
            if best_score_found:  # should be always true
                result.append(best_score * best_scoring_len)
        return sum(result)
