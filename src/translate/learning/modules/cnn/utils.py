import math
from translate.backend.utils import backend


class ResBlock(backend.nn.Module):
    """
        Note To Self:  using padding to "mask" the convolution is equivalent to
        either centering the convolution (no mask) or skewing the convolution to
        the left (mask).  Either way, we should end up with n timesteps.
        Also note that "masked convolution" and "casual convolution" are two
        names for the same thing.
    Args:
        d (int): size of inner track of network.
        r (int): size of dilation
        k (int): size of kernel in dilated convolution (odd numbers only)
        casual (bool): determines how to pad the casual conv layer. See notes.
    """

    def __init__(self, d, r=1, k=3, casual=False, use_bias=False):
        super(ResBlock, self).__init__()
        self.d = d  # input features
        self.r = r  # dilation size
        self.k = k  # "masked kernel size"
        ub = use_bias
        self.layernorm1 = backend.nn.InstanceNorm1d(num_features=2 * d, affine=True)  # same as LayerNorm
        self.relu1 = backend.nn.ReLU(inplace=True)
        self.conv1x1_1 = backend.nn.Conv1d(2 * d, d, kernel_size=1, bias=ub)  # output is "d"
        self.layernorm2 = backend.nn.InstanceNorm1d(num_features=d, affine=True)
        self.relu2 = backend.nn.ReLU(inplace=True)
        if casual:
            padding = (self._same_pad(k, r), 0)
        else:
            p = self._same_pad(k, r)
            if p % 2 == 1:
                padding = [p // 2 + 1, p // 2]
            else:
                padding = (p // 2, p // 2)
        self.pad = backend.nn.ConstantPad1d(padding, 0.)
        # self.pad = nn.ReflectionPad1d(padding) # this might be better for audio
        self.maskedconv1xk = backend.nn.Conv1d(d, d, kernel_size=k, dilation=r, bias=ub)
        self.layernorm3 = backend.nn.InstanceNorm1d(num_features=d, affine=True)
        self.relu3 = backend.nn.ReLU(inplace=True)
        self.conv1x1_2 = backend.nn.Conv1d(d, 2 * d, kernel_size=1, bias=ub)  # output is "2*d"

    def forward(self, input_):
        x = input_
        x = self.layernorm1(x)
        x = self.relu1(x)
        x = self.conv1x1_1(x)
        x = self.layernorm2(x)
        x = self.relu2(x)
        x = self.pad(x)
        x = self.maskedconv1xk(x)
        x = self.layernorm3(x)
        x = self.relu3(x)
        x = self.conv1x1_2(x)
        # print("ResBlock:", x.size(), input.size())
        x += input_  # add back in residual
        return x

    @staticmethod
    def _same_pad(k=1, dil=1):
        # assumes stride length of 1
        # p = math.ceil((l - 1) * s - l + dil*(k - 1) + 1)
        p = math.ceil(dil * (k - 1))
        # print("padding:", p)
        return p


class ResBlockSet(backend.nn.Module):
    """
        The Bytenet encoder and decoder are made up of sets of residual blocks
        with dilations of increasing size.  These sets are then stacked upon each
        other to create the full network.
    """

    def __init__(self, d, max_r=16, k=3, casual=False):
        super(ResBlockSet, self).__init__()
        self.d = d
        self.max_r = max_r
        self.k = k
        rlist = [1 << x for x in range(15) if (1 << x) <= max_r]
        self.blocks = backend.nn.Sequential(*[ResBlock(d, r, k, casual) for r in rlist])

    def forward(self, input_):
        x = input_
        x = self.blocks(x)
        return x
