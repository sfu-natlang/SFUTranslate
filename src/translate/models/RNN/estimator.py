"""
Provides the optimizer creation, loss computation, back-propagation, and scoring functionalities
 on the input batches for structured prediction training tasks in which the input is composed of at least two sequences
"""
from typing import Callable, Iterable, Tuple, List, Type

from translate.configs.loader import ConfigLoader
from translate.models.abs.modelling import AbsCompleteModel
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
        self._eps = 7. / 3. - 4. / 3. - 1.

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


class SPEstimator:
    def __init__(self, configs: ConfigLoader, model: Type[AbsCompleteModel],
                 compute_score_function: Type[Callable[[Iterable[Iterable[int]],
                                                        Iterable[Iterable[int]], bool, bool], float]]):
        """

        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param model: the sequence to sequence model instance object which will be used for computing the model
         predictions and parameter optimization
        :param compute_score_function: the function handler passed from the dataset to be called for converting back the
         lists (tensors) of ids to target sentences and computing the average bleu score on them.
        """
        self.optim_name = configs.get("trainer.optimizer.name", must_exist=True)
        self.learning_rate = float(configs.get("trainer.optimizer.lr", must_exist=True))
        self.grad_clip_norm = configs.get("trainer.optimizer.gcn", 5)
        self.model = model
        self.optimizers = [create_optimizer(self.optim_name, x, lr=self.learning_rate)
                           for x in model.optimizable_params_list()]
        self.grad_stats = RunStats()
        self.no_grad_stats = RunStats()
        self.compute_score = compute_score_function

    def step(self, input_tensor_batch: backend.Tensor, target_tensor_batch: backend.Tensor, *args, **kwargs) \
            -> Tuple[float, List[List[int]]]:
        """
        The step function which takes care of computing the loss, gradients and back-propagating them given
         the input (:param input_tensor_batch:), output (:param target_tensor_batch:) pair of batch tensors.
        :return: the average loss value for the batch instances plus the decoded output computed over the batch
        """
        for opt in self.optimizers:
            opt.zero_grad()
        _loss_, _loss_size_, computed_output = self.model.forward(input_tensor_batch, target_tensor_batch, args, kwargs)
        _loss_.backward()
        if self.grad_clip_norm > 0.0:
            [backend.nn.utils.clip_grad_norm_(x, self.grad_clip_norm) for x in self.model.optimizable_params_list()]
        loss_value = _loss_.item() / _loss_size_
        self.grad_stats.update(0.0, loss_value)
        for opt in self.optimizers:
            opt.step()
        return loss_value, computed_output

    def step_no_grad(self, input_tensor_batch: backend.Tensor, target_tensor_batch: backend.Tensor, *args, **kwargs):
        """
        The function which given a pair of input (:param input_tensor_batch:), and output (:param target_tensor_batch:)
         tensors, freezes the model parameters then computes the model loss over its predictions and updates
          the relevant stat values.
        :returns a sample pair of reference, prediction sentences for logging purposes (can be ignored!)
        """
        with backend.no_grad():
            _loss_, _loss_size_, computed_output = self.model.forward(input_tensor_batch, target_tensor_batch,
                                                                      args, kwargs)
            loss_value = _loss_.item() / _loss_size_
            score, ref_sample, hyp_sample = self.compute_score(target_tensor_batch, computed_output, ref_is_tensor=True)
            self.no_grad_stats.update(score, loss_value)
            return ref_sample, hyp_sample

    def __str__(self):
        return "[TL {:.3f} DL {:.3f} DS {:.3f}]".format(
            self.grad_stats.loss, self.no_grad_stats.loss, self.no_grad_stats.bscore)
