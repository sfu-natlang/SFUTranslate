"""
The implementation of the Decoder module (containing the Residual Blocks) in ByteNet framework
"""
from translate.backend.utils import backend
from translate.learning.modules.cnn.utils import ResBlockSet

__author__ = "Hassan S. Shavarani"


class CharCNNDecoder(backend.nn.Module):
    def __init__(self, d=512, max_r=16, k=3, num_sets=6, num_classes=205,
                 reduce_out=None, use_logsm=True):
        """
        :param d: size of hidden units in 1D convolutional layers
        :param max_r: maximum dilation rate (paper default: 16)
        :param k: masked kernel size (paper default: 3)
        :param num_sets: number of residual sets (paper default: 6. 5x6 = 30 ResBlocks)
        :param num_classes: number of output classes (Hunter prize default: 205)
         This parameter needs to be set to the size of the vocabulary (in character level data settings)
        :param reduce_out: if set another convolutional layer is added to each Redidual Block Set to reduce the size of
         output passed to the generator convolutional layers
        :param use_logsm: the flag stating whether the output needs to be passed through a softmax layer
        """
        super(CharCNNDecoder, self).__init__()
        self.max_r = max_r
        self.k = k
        self.d = d
        self.num_sets = num_sets
        self.use_logsm = use_logsm  # this is for NLLLoss
        self.sets = backend.nn.Sequential()
        for i in range(num_sets):
            self.sets.add_module("set_{}".format(i + 1), ResBlockSet(d, max_r, k, True))
            if reduce_out is not None:
                r = reduce_out[i]
                if r != 0:
                    reduce_conv = backend.nn.Conv1d(2 * d, 2 * d, r, r)
                    reduce_pad = backend.nn.ConstantPad1d((0, r), 0.)
                    self.sets.add_module("reduce_pad_{}".format(i + 1), reduce_pad)
                    self.sets.add_module("reduce_{}".format(i + 1), reduce_conv)
        self.conv1 = backend.nn.Conv1d(2 * d, 2 * d, 1)
        self.conv2 = backend.nn.Conv1d(2 * d, num_classes, 1)
        self.logsm = backend.nn.LogSoftmax(dim=1)

    def forward(self, input_):
        x = input_
        x = self.sets(x)
        x = self.conv1(x)
        x = backend.nn.functional.relu(x)
        x = self.conv2(x)
        if self.use_logsm:
            x = self.logsm(x)
        return x
