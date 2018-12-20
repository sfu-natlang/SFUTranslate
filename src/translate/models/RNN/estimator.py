"""
Provides the optimizer creation, loss computation, back-propagation, and scoring functionalities
 on the single input batches
"""
from typing import Callable, Iterable, Tuple, List

from translate.configs.loader import ConfigLoader
from translate.models.RNN.seq2seq import SequenceToSequence
from translate.models.backend.utils import backend

__author__ = "Hassan S. Shavarani"


def create_optimizer(optimizer_name, unfiltered_params, lr):
    """
    The method to create the optimizer object given the desired optimizer name (:param optimizer_name:) and the expected
     learning late (:param lr:) for the set of model parameters (:param unfiltered_params:)
    :return: the created optimizer object
    :raises ValueError if the requested optimizer name is not defined
    """
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
    """
    The loss, score, and result size container, used for storing the run stats of the training/testing iterations
    """
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
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param model: the sequence to sequence model instance object which will be used for computing the model
         predictions and parameter optimization
        :param compute_bleu_function: the function handler passed from the dataset to be called for converting back the
         lists (tensors) of ids to target sentences and computing the average bleu score on them.
        """
        self.optim_name = configs.get("trainer.optimizer.name", must_exist=True)
        self.learning_rate = float(configs.get("trainer.optimizer.lr", must_exist=True))
        self.grad_clip_norm = configs.get("trainer.optimizer.gcn", 5)
        self.model = model
        self.encoder_optimizer = create_optimizer(self.optim_name, model.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = create_optimizer(self.optim_name, model.decoder.parameters(), lr=self.learning_rate)
        self.train_stats = RunStats()
        self.dev_stats = RunStats()
        self.compute_bleu = compute_bleu_function

    def step(self, input_tensor_batch: backend.Tensor, target_tensor_batch: backend.Tensor) -> Tuple[float, List[List[int]]]:
        """
        The step function which takes care of computing the loss, gradients and back-propagating them given
         the input (:param input_tensor_batch:), output (:param target_tensor_batch:) pair of batch tensors.
        :return: the average loss value for the batch instances plus the decoded output computed over the batch
        """
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
        """
        The function which given a pair of input (:param input_tensor_batch:), and output (:param target_tensor_batch:)
         tensores, freezes the model parameters then computes the model loss over its predictions and updates
          the relevant stat values.
        :returns a sample pair of reference, prediction sentences for logging purposes (can be ignored!)
        """
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