"""
Implementation of Label Smoothing Regularization as explained in:
 https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf
"""
from translate.backend.utils import backend, Variable

__author__ = "Hassan S. Shavarani"


class LabelSmoothing(backend.nn.Module):
    """
    This class can be used directly in replacement of NLLLoss of KLDivLoss criterions.
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        :param size: the target space vocabulary size 
        :param padding_idx: the word index of padding token
        :param smoothing: the smoothing probability applied to the target distribution
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = backend.nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = backend.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0 and len(mask) > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))
