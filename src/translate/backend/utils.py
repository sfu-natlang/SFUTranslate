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
    The function which copies the tensor alongside its one to the last dimension (assumed to be the batch dimension)
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


def row_wise_batch_split(input_tensor, split_size):
    """
    The function which splits the tensor alongside its one to the last dimension (assumed to be the batch dimension)
    :param input_tensor: the tensor to be copied
    :param split_size: the number of splits alongside each tensor
    :return: the array containing the splitted tensors
    """
    single_dimensional = False
    if input_tensor.dim() == 1:
        input_tensor = input_tensor.unsqueeze(-1)
        single_dimensional = True
    if input_tensor.dim() == 2 and single_dimensional:
        assert input_tensor.size(0) % split_size == 0
        output_tensors = [input_tensor[i:input_tensor.size(0):split_size].squeeze(-1) for i in range(split_size)]
    elif input_tensor.dim() == 2:
        assert input_tensor.size(0) % split_size == 0
        output_tensors = [input_tensor[i:input_tensor.size(0):split_size] for i in range(split_size)]
    elif input_tensor.dim() == 3:
        assert input_tensor.size(1) % split_size == 0
        output_tensors = [input_tensor[:, i:input_tensor.size(1):split_size] for i in range(split_size)]
    elif input_tensor.dim() == 4:
        assert input_tensor.size(2) % split_size == 0
        output_tensors = [input_tensor[:, :, i:input_tensor.size(2):split_size] for i in range(split_size)]
    else:
        raise ValueError("Tensor of dimensions higher than 4 are not supported in this version!")
    return output_tensors


def recombine_tensors_to_a_batch(input_tensors_list):
    for elem in input_tensors_list:
        assert input_tensors_list[0].size() == elem.size()
    batch_size = len(input_tensors_list)
    tensor_sizes = list(input_tensors_list[0].size())

    if input_tensors_list[0].dim() == 1:
        input_tensors_list = [elem.unsqueeze(-1) for elem in input_tensors_list]
        result = backend.cat(input_tensors_list, dim=-1).view(-1)
    else:
        tensor_sizes[-2] *= batch_size
        result = backend.cat(input_tensors_list, dim=-1).view(*tensor_sizes)
    return result


def __test_copy_split(input_tensor):
    a1 = row_wise_batch_copy(input_tensor, 5)
    a2 = row_wise_batch_split(a1, 5)
    a3 = recombine_tensors_to_a_batch(a2)
    print(a1.size(), len(a2), a2[0].size(), a3.size())
    assert (a1 == a3).sum().item() == (a1 == a1).sum().item()

