"""
The backend utils function which is the only class accessing directly to torch library. The renaming of torch to backend
 has been performed for the sake of traceability; later, we can simply look for occurrences of "backend" and change it
  to tf or any backend we may need!

This script contains any conversion functions which are necessary for limiting the direct access of project classes to
 the torch library itself
"""
import torch as backend
from torch.utils.data import DataLoader
from torch.autograd import Variable

device = backend.device("cuda" if backend.cuda.is_available() else "cpu")

__author__ = "Hassan S. Shavarani"


def zeros_tensor(d1, d2, d3):
    return backend.zeros(d1, d2, d3).to(device)


def list_to_long_tensor(list_long_values):
    return backend.LongTensor(list_long_values).to(device)


def long_tensor(d1, d2, d3):
    return backend.LongTensor(d1, d2, d3).to(device)


def tensor2list(tensor: backend.Tensor):
    return tensor.cpu().numpy().tolist()


def row_wise_batch_copy(input_tensor, copy_size):
    """
    The function which copies the tensor alongside its one to last dimension
    :param input_tensor: the tensor to be copied
    :param copy_size: the number of times each tensor gets copied
    :return: the duplicated tensor
    """
    single_dimensional = False
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(-1)
        single_dimensional = True
    sizes = [1] * input_tensor.dim()
    sizes[-1] = copy_size
    view_sizes = list(input_tensor.size())
    view_sizes[-2] *= copy_size
    output = input_tensor.repeat(*sizes).view(*view_sizes)
    if single_dimensional:
        return output.squeeze(-1)
    else:
        return output
