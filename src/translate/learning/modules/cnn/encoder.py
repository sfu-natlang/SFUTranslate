"""
The implementation of the Encoder module (containing the Residual Blocks) in ByteNet framework
"""
from translate.backend.utils import backend

from translate.learning.modules.cnn.utils import ResBlockSet

__author__ = "Hassan S. Shavarani"


class CharCNNEncoder(backend.nn.Module):
    def __init__(self, d=800, max_r=16, k=3, num_sets=6):
        """
        :param d: size of hidden units in 1D convolutional layers
        :param max_r: maximum dilation rate (paper default: 16)
        :param k: masked kernel size (paper default: 3)
        :param num_sets: number of residual sets (paper default: 6. 5x6 = 30 ResBlocks)
        """
        super(CharCNNEncoder, self).__init__()
        self.d = d
        self.max_r = max_r
        self.k = k
        self.num_sets = num_sets
        self.pad_in = backend.nn.ConstantPad1d((0, 1), 0.)
        self.conv_in = backend.nn.Conv1d(1, 2 * d, 1)
        self.sets = backend.nn.Sequential()
        for i in range(num_sets):
            self.sets.add_module("set_{}".format(i + 1), ResBlockSet(d, max_r, k))
        self.conv_out = backend.nn.Conv1d(2 * d, 2 * d, 1)

    def forward(self, input_):
        x = input_
        x = self.conv_in(x)
        x = self.sets(x)
        x = self.conv_out(x)
        x = backend.nn.functional.relu(x)
        return x
