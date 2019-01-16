from translate.backend.utils import backend
from translate.learning.modules.cnn.utils import ResBlockSet


class CharCNNDecoder(backend.nn.Module):
    """
        d = hidden units
        max_r = maximum dilation rate (paper default: 16)
        k = masked kernel size (paper default: 3)
        num_sets = number of residual sets (paper default: 6. 5x6 = 30 ResBlocks)
        num_classes = number of output classes (Hunter prize default: 205)
    """

    def __init__(self, d=512, max_r=16, k=3, num_sets=6, num_classes=205,
                 reduce_out=None, use_logsm=True):
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

    def generate(self, input_, n_samples, encoder=None):
        bs = input_.size(0)
        x = input_
        for i in range(n_samples):
            out = self(x)
            if i + 1 != n_samples:
                gen_next = out.max(1)[1].index_select(1, out.new([out.size(2) - 1]).long())
                gen_enc = encoder(gen_next)
                x = backend.cat((x, gen_enc), dim=2)
        # add last generated output to out
        gen_last = out.index_select(2, out.new([out.size(2) - 1]).long())
        out = backend.cat((out, gen_last), dim=2)
        # return only generated outputs
        tot_samples = out.size(2)
        out = out.index_select(2, out.new(range(tot_samples - n_samples, tot_samples)).long())
        return out
