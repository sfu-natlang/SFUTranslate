"""
Provides an interface for a completed model, which is of Type backen.nn.Module, an will be able to compute the loss
 value for a given input, in addition to the final output.
"""
from typing import Type, List, Tuple, Any
from translate.backend.utils import backend

from abc import ABC, abstractmethod

__author__ = "Hassan S. Shavarani"


class AbsCompleteModel(backend.nn.Module, ABC):
    def __init__(self, criterion: Type[backend.nn.Module]):
        """
        :param criterion: the Loss instance which will be used in computation of the final gradient value 
        """
        super(AbsCompleteModel, self).__init__()
        self.criterion = criterion

    def forward(self, input_tensor: backend.Tensor, target_tensor: backend.Tensor, *args, **kwargs) \
            -> Tuple[backend.Tensor, int, List[Any]]:
        """
        The Module forward function which will compute the following three values and returns them.
         1. the final model prediction using the input values passed tp the function and compares it with expected 
          values passed to itself to compute the prediction loss (using self.criterion) as the first return type
         2. the length of the target prediction (could be different from the passed passed expected values)
         3. the final list of batch predictions (a list of BatchSize containing the predictions for each batch)
        """
        raise NotImplementedError

    @abstractmethod
    def optimizable_params_list(self) -> List[Any]:
        """
        :return: a list of sets of parameters which need to be optimized separately (e.g. in an encoder-decoder model,
         you may return a list of size two, the first of which containing parameters of the encoder and the second, 
          containing the parameters of the decoder  
        """
        raise NotImplementedError

    @abstractmethod
    def validate_instance(self, *args, **kwargs) -> Tuple[float, str]:
        """
        The function in charge of validation of model prediction results (could be based on some expected values), and
         returning the validation score in addition to a sample of the prediction.
        :return: the computed validation score in addition to a sample result
        """
        raise NotImplementedError
