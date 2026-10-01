# Code adapted from the RevIN implementation by ts-kim:
# https://github.com/ts-kim/RevIN
#
# Original license applies.
# Minor modifications were made for integration into this project.


import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str, channel_idx=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x, channel_idx)
        elif mode == 'denorm':
            x = self._denormalize(x, channel_idx)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x, channel_idx=None):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            weight = self.affine_weight if channel_idx is None else self.affine_weight[channel_idx]
            bias = self.affine_bias if channel_idx is None else self.affine_bias[channel_idx]
            x = x * weight
            x = x + bias
        return x

    def _denormalize(self, x, channel_idx=None):
        if self.affine:
            weight = self.affine_weight if channel_idx is None else self.affine_weight[channel_idx]
            bias = self.affine_bias if channel_idx is None else self.affine_bias[channel_idx]
            x = x - bias
            x = x / (weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
