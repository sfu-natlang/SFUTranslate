"""
Provides the optimizer creation, loss computation, back-propagation, and scoring functionalities
 on the input batches for structured prediction training tasks in which the input is composed of at least two sequences
"""
from typing import Tuple, List, Type

from translate.backend.utils import backend
from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
from translate.readers.constants import ReaderType

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


class StatCollector:
    """
    The loss, score, and result size container, used for storing the run stats of the training/testing iterations
    """

    def __init__(self):
        self._eps = 7. / 3. - 4. / 3. - 1.
        self._test_total = 0.0
        self._test_score = 0.0
        self._test_loss = 0.0
        self._dev_total = 0.0
        self._dev_score = 0.0
        self._dev_loss = 0.0
        self._train_total = 0.0
        self._train_score = 0.0
        self._train_loss = 0.0

    @property
    def test_score(self) -> float:
        return self._test_score / (self._test_total + self._eps)

    @property
    def test_loss(self) -> float:
        return self._test_loss / (self._test_total + self._eps)

    @property
    def dev_score(self) -> float:
        return self._dev_score / (self._dev_total + self._eps)

    @property
    def dev_loss(self) -> float:
        return self._dev_loss / (self._dev_total + self._eps)

    @property
    def train_score(self) -> float:
        return self._train_score / (self._train_total + self._eps)

    @property
    def train_loss(self) -> float:
        return self._train_loss / (self._train_total + self._eps)

    def update(self, score: float, loss: float, stat_type: ReaderType):
        if stat_type == ReaderType.TRAIN:
            self._train_score += score
            self._train_loss += loss
            self._train_total += 1.0
        elif stat_type == ReaderType.TEST:
            self._test_score += score
            self._test_loss += loss
            self._test_total += 1.0
        elif stat_type == ReaderType.DEV:
            self._dev_score += score
            self._dev_loss += loss
            self._dev_total += 1.0
        else:
            raise NotImplementedError


class Estimator:
    def __init__(self, configs: ConfigLoader, model: Type[AbsCompleteModel]):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param model: the sequence to sequence model instance object which will be used for computing the model
         predictions and parameter optimization
        """
        self.optim_name = configs.get("trainer.optimizer.name", must_exist=True)
        self.learning_rate = float(configs.get("trainer.optimizer.lr", must_exist=True))
        self.grad_clip_norm = configs.get("trainer.optimizer.gcn", 5)
        self.model = model
        self.optimizers = [create_optimizer(self.optim_name, x, lr=self.learning_rate)
                           for x in model.optimizable_params_list()]

    def step(self, *args, **kwargs) -> Tuple[float, List[List[int]]]:
        """
        The step function which takes care of computing the loss, gradients and back-propagating them given
         the input tensors (the number of them could vary based on the application).
        :return: the average loss value for the batch instances plus the decoded output computed over the batch
        """
        for opt in self.optimizers:
            opt.zero_grad()
        _loss_, _loss_size_, computed_output = self.model.forward(*args, *kwargs)
        _loss_.backward()
        if self.grad_clip_norm > 0.0:
            [backend.nn.utils.clip_grad_norm_(x, self.grad_clip_norm) for x in self.model.optimizable_params_list()]
        loss_value = _loss_.item() / _loss_size_
        for opt in self.optimizers:
            opt.step()
        return loss_value, computed_output

    def step_no_grad(self, *args, **kwargs):
        """
        The function which given a pair of input tensors, freezes the model parameters then computes the model loss over
         its predictions.
        :return: the average loss value for the batch instances plus the decoded output computed over the batch
        """
        with backend.no_grad():
            _loss_, _loss_size_, computed_output = self.model.forward(*args, *kwargs)
            loss_value = _loss_.item() / _loss_size_
            return loss_value, computed_output
