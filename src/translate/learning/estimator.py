"""
Provides the optimizer creation, loss computation, back-propagation, and scoring functionalities
 on the input batches for structured prediction training tasks in which the input is composed of at least two sequences
"""
import os
import math
from typing import Tuple, List, Type

from translate.backend.utils import backend
from translate.configs.loader import ConfigLoader
from translate.learning.modelling import AbsCompleteModel
from translate.readers.constants import ReaderType
from translate.logging.utils import logger

__author__ = "Hassan S. Shavarani"


def create_optimizer(optimizer_name, unfiltered_params, lr, warmup_wrapper_needed=False, configs=None):
    """
    The method to create the optimizer object given the desired optimizer name (:param optimizer_name:) and the expected
     learning late (:param lr:) for the set of model parameters (:param unfiltered_params:)
     In case the learning rate warmup wrapper is required you can set the :param  warmup_wrapper_needed: to True and 
      pass the configs object for the wrapper to get configured. 
    :return: the created optimizer object
    :raises ValueError if the requested optimizer name is not defined
    """
    if warmup_wrapper_needed:
        # the learning rate would gradually increase during the warmup
        lr = 0.0
    params = filter(lambda x: x.requires_grad, unfiltered_params)
    if optimizer_name == "adam":
        optim = backend.optim.Adam(params, lr=lr, betas=(0.9, 0.98), eps=1e-9)
    elif optimizer_name == "adadelta":
        optim = backend.optim.Adadelta(params, lr=lr)
    elif optimizer_name == "sgd":
        optim = backend.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError("No optimiser found with the name {}".format(optimizer_name))
    if not warmup_wrapper_needed:
        return optim
    else:
        return OptimizerWrapperWithWarmUpSteps(configs, optim)


def create_scheduler(scheduler_name, optimizer, configs):
    if scheduler_name.lower() == "cosine":
        n_epochs = configs.get("trainer.optimizer.epochs", must_exist=True)
        n_epochs = n_epochs if n_epochs > 0 else 1
        eta_min = float(configs.get("trainer.optimizer.scheduler.eta_min", must_exist=True))
        return backend.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min)
    elif scheduler_name.lower() == "step":
        step_size = configs.get("trainer.optimizer.scheduler.step_size", must_exist=True)
        gamma = float(configs.get("trainer.optimizer.scheduler.gamma", must_exist=True))
        return backend.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise NotImplementedError


class StatCollector:
    """
    The loss, score, and result size container, used for storing the run stats of the training/testing iterations
    """

    def __init__(self, train_size, model_batch_size, higher_score_is_better):
        self._eps = 7. / 3. - 4. / 3. - 1.
        self._higher_score_is_better = higher_score_is_better
        self._test_total = 0.0
        self._test_score = 0.0
        self._test_loss = 0.0
        self._dev_total = 0.0
        self._dev_score = 0.0
        self._dev_loss = 0.0
        self._train_total = 0.0
        self._train_score = 0.0
        self._train_loss = 0.0

        self._best_train_loss = float('+inf')
        self._best_dev_loss = float('+inf')
        self._best_dev_score = float('-inf') if higher_score_is_better else float('+inf')
        self.global_step = 0.0

        # the value which is used for performing the dev set evaluation steps
        self._trainset_size = train_size
        self._training_batch_size = model_batch_size
        self._print_every = math.ceil(0.25 * int(math.ceil(float(train_size) / float(model_batch_size))))
        self._train_iter_step = 0.0

    def zero_step(self):
        self._train_iter_step = 0.0
        self._train_score = 0.0
        self._train_loss = 0.0
        self._train_total = 0.0

    def step(self):
        self._train_iter_step += 1.0

    def validation_required(self):
        return self._train_iter_step % self._print_every == 0

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

    def reset(self, stat_type: ReaderType):
        if stat_type == ReaderType.TRAIN:
            self._train_score = 0.0
            self._train_loss = 0.0
            self._train_total = 0.0
        elif stat_type == ReaderType.TEST:
            self._test_score = 0.0
            self._test_loss = 0.0
            self._test_total = 0.0
        elif stat_type == ReaderType.DEV:
            self._dev_score = 0.0
            self._dev_loss = 0.0
            self._dev_total = 0.0
        else:
            raise NotImplementedError

    def improved_recently(self) -> bool:
        """
        Checks whether the stat collector has seen any loss improvements from the last time it was asked
        """
        improved = False
        self.global_step += 1.0
        if self.dev_loss < self._best_dev_loss:
            self._best_dev_loss = self.dev_loss
            improved = True
        if self.train_loss < self._best_train_loss:
            self._best_train_loss = self.train_loss
            # improved = True
        if not self._higher_score_is_better and self.dev_score < self._best_dev_score:
            self._best_dev_score = self.dev_score
            improved = True
        if self._higher_score_is_better and self.dev_score > self._best_dev_score:
            self._best_dev_score = self.dev_score
            improved = True
        return improved


class OptimizerWrapperWithWarmUpSteps:
    def __init__(self, configs: ConfigLoader, optimizer_instance: Type[backend.optim.Optimizer]):
        """
        :param configs: an instance of ConfigLoader which has been loaded with a yaml config file
        :param optimizer_instance: The optimizer instance the learning rate of which is supposed to be updated with 
         after every single loss backward step. 
        """
        super(OptimizerWrapperWithWarmUpSteps, self).__init__()
        self.optimizer = optimizer_instance
        self._step = 0
        # the number of warmup steps
        self.warmup = configs.get("trainer.optimizer.warmup_steps", must_exist=True)
        # the rate update factor used in learning rate update computation
        self.factor = configs.get("trainer.optimizer.lr_update_factor", must_exist=True)
        # the size of the source embedding layer of the model
        # TODO the value must be able to be taken from a non-transformer model as well!
        self.model_size = configs.get("trainer.optimizer.d_model", must_exist=True)
        self._rate = 0
        logger.info("Optimizer loaded into the learning rate warmup wrapper for the model size: {} with {} warm-up "
                    "states and the learning rate updates of factor {}".format(
            self.model_size, self.warmup, self.factor))

    def step(self):
        """
        Performs a single optimization step with the warmup strategy stated in the "attention is all you need" paper.
        """
        self._step += 1
        rate = self._rate_()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _rate_(self, step=None):
        """
        Class method in charge of updating the learning rate considering the :param step: number passed to it.
         The update will be performed according to the "lrate" formula (Equation 3, Page 7 of the "attention is all you 
          need") paper. 
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


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
        warmup_needed = configs.get("trainer.optimizer.needs_warmup", False)
        self.experiment_name = configs.get("trainer.experiment.name", "unnamed")
        self.model = model
        logger.info('Loading {} optimizer(s) of type \"{}\" for training the model'.format(
            len(model.optimizable_params_list()), self.optim_name.upper()))
        self.optimizers = [create_optimizer(self.optim_name, x, self.learning_rate, warmup_needed, configs)
                           for x in model.optimizable_params_list()]
        if configs.get("trainer.optimizer.scheduler", None) is not None and not warmup_needed:
            self.scheduler_name = configs.get("trainer.optimizer.scheduler.name", must_exist=True)
            self.schedulers = [create_scheduler(self.scheduler_name, opt, configs) for opt in self.optimizers]
        else:
            self.scheduler_name = None
            self.schedulers = []

    def step_schedulers(self):
        if len(self.schedulers):
            logger.info("Updating the learning rates through {} scheduler ...".format(self.scheduler_name))
        for scheduler in self.schedulers:
            scheduler.step()

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
        loss_value = _loss_.item() / _loss_size_ if _loss_size_ > 0.0 else 0.0
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
            loss_value = _loss_.item() / _loss_size_ if _loss_size_ > 0.0 else 0.0
            return loss_value, computed_output

    def save_checkpoint(self, stat_collector: StatCollector) -> str:
        """
        Saves the model and returns the saved checkpoint address, the function uses the collected stats during training 
         to form the saving checkpoint address
        """
        checkpoint = {'global_step': stat_collector.global_step, 'model_state_dict': self.model.state_dict()}
        checkpoint_path = 'checkpoints/%s_acc_%.2f_loss_%.2f_step_%d.pt' % (
            self.experiment_name, stat_collector.dev_score, stat_collector.dev_loss, stat_collector.global_step)
        directory, filename = os.path.split(os.path.abspath(checkpoint_path))
        if not os.path.exists(directory):
            os.makedirs(directory)
        backend.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> Type[AbsCompleteModel]:
        """
        Loades the model from the :param checkpoint_path: and returns the loaded model object 
        """
        # It's weird that if `map_location` is not given, it will be extremely slow.
        ckpt = backend.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(ckpt['model_state_dict'])
        return self.model
