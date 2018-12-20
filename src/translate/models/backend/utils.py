import torch as backend  # renamed for the sake of traceability
# (later, we can simply look for occurrences of "backend" and change it to tf or anything we need!)
from torch.utils.data import DataLoader
from torch.autograd import Variable
device = backend.device("cuda" if backend.cuda.is_available() else "cpu")


def zeros_tensor(d1, d2, d3):
    return backend.zeros(d1, d2, d3).to(device)


def list_to_long_tensor(list_long_values):
    return backend.LongTensor(list_long_values).to(device)


def long_tensor(d1, d2, d3):
    return backend.LongTensor(d1, d2, d3).to(device)


def tensor2list(tensor: backend.Tensor):
    return tensor.cpu().numpy().tolist()