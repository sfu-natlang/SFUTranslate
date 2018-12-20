from typing import Callable, Iterable

from translate.configs.loader import ConfigLoader
from translate.models.RNN.seq2seq import SequenceToSequence
from translate.models.backend.utils import backend


def create_optimizer(optimizer_name, unfiltered_params, lr):
    params = filter(lambda x: x.requires_grad, unfiltered_params)
    if optimizer_name == "adam":
        return backend.optim.Adam(params, lr=lr)
    elif optimizer_name == "adadelta":
        return backend.optim.Adadelta(params, lr=lr)
    elif optimizer_name == "sgd":
        return backend.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError("No optimiser found with the name {}".format(optimizer_name))


class RunStats:
    def __init__(self):
        self._total = 0.0
        self._score = 0.0
        self._loss = 0.0
        self._eps = 7./3. - 4./3. - 1.

    @property
    def bscore(self) -> float:
        return self._score / (self._total + self._eps)

    @property
    def loss(self) -> float:
        return self._loss / (self._total + self._eps)

    def update(self, score: float, loss: float):
        self._score += score
        self._loss += loss
        self._total += 1.0


class STSEstimator:
    def __init__(self, configs: ConfigLoader, model: SequenceToSequence,
                 compute_bleu_function: Callable[[Iterable[Iterable[int]], Iterable[Iterable[int]], bool, bool], float]):
        self.optim_name = configs.get("trainer.optimizer.name", must_exist=True)
        self.learning_rate = float(configs.get("trainer.optimizer.lr", must_exist=True))
        self.grad_clip_norm = configs.get("trainer.optimizer.gcn", 5)
        self.model = model
        self.encoder_optimizer = create_optimizer(self.optim_name, model.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = create_optimizer(self.optim_name, model.decoder.parameters(), lr=self.learning_rate)
        self.train_stats = RunStats()
        self.dev_stats = RunStats()
        self.compute_bleu = compute_bleu_function

    def step(self, input_tensor_batch: backend.Tensor, target_tensor_batch: backend.Tensor):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        batch_loss, batch_loss_size, decoded_word_ids = self.model.forward(input_tensor_batch, target_tensor_batch)
        batch_loss.backward()
        if self.grad_clip_norm > 0.0:
            backend.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.grad_clip_norm)
            backend.nn.utils.clip_grad_norm_(self.model.decoder.parameters(), self.grad_clip_norm)
        loss_value = batch_loss.item() / batch_loss_size
        self.train_stats.update(0.0, loss_value)
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return loss_value, decoded_word_ids

    def step_no_grad(self, input_tensor_batch: backend.Tensor, target_tensor_batch: backend.Tensor):
        with backend.no_grad():
            batch_loss, batch_loss_size, decoded_word_ids = self.model.forward(input_tensor_batch, target_tensor_batch)
            loss_value = batch_loss.item() / batch_loss_size
            bleu_score, ref_sample, hyp_sample = self.compute_bleu(target_tensor_batch,
                                                                   decoded_word_ids, ref_is_tensor=True)
            self.dev_stats.update(bleu_score, loss_value)
            return ref_sample, hyp_sample

    def __str__(self):
        return "[TL {:.3f} DL {:.3f} DS {:.3f}]".format(
            self.train_stats.loss, self.dev_stats.loss, self.dev_stats.bscore)