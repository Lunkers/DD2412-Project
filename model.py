import torch
import torch.nn as nn
import numpy as np
from affinecoupling import AffineCouplingLayer
from actnorm import ActNorm
from invconv import InvConv
from math import log, pi
from torch.nn import functional as F
from zeroconv import ZeroConv2d


# TODO: Make sure dimens are correct through the operations
# TODO: Check that it trains correctly and optimizes the right parameters (make sure nn.ModuleList works correctly)
# TODO: Make sure actnorm, inconv and affine coupling are complete and correct
# TODO: Original code uses same padding, but there isn't a corresponding one in pytorch, wonder if we have to take into consideration?

class Glow(nn.Module):
    def __init__(self, in_channels, levels, depth):
        super(Glow, self).__init__()
        # expecting shape of (batch_size, channels, height, width)
        self.levels = levels
        self.depth = depth
        self.blocks = nn.ModuleList()
        self.zeroconv = nn.ModuleList()
        self.initialize_blocks(in_channels)
        self.initalize_zeroconv(in_channels)

    def initialize_blocks(self, in_channels):
        for _ in range(self.levels):
            self.blocks.append(Block(in_channels, self.depth))
            # Note that at the last iteration the in_channel will not be applied to blocks, because split is not applied
            in_channels *= 2  # because we apply a split at the end of each level iteration and split has squeeze in it as well resulting in a factor of 2

    def initalize_zeroconv(self, in_channels):
        for _ in range(self.levels - 1):
            self.zeroconv.append(ZeroConv2d(in_channels * 2, in_channels * 4))
            in_channels *= 2
        self.zeroconv.append(ZeroConv2d(in_channels * 4, in_channels * 8))

    def forward(self, x):
        logdet = 0
        eps = []
        logp_sum = 0
        for i, current_block in enumerate(self.blocks):
            x = squeeze(x)
            x, logds = current_block(x)
            logdet = logdet + logds
            # print(f"logdet: {logdet.shape}\nx: {x.shape}")
            if i < self.levels - 1: #last block shouldnt split
                x, _eps, logp = split(x, self.zeroconv[i])
                # print(f"after split\niter: {i}\nlogdet: {logdet.shape}\nx: {x.shape}")
                eps.append(_eps)
            elif i == len(self.blocks) - 1:  # perform this at last iteration
                zero = torch.zeros_like(x)
                mean, log_sd = self.zeroconv[i](zero).chunk(2, 1)
                logp = gaussian_log_p(x, mean, log_sd)
                logp = logp.sum(dim=[1, 2, 3])
            logp_sum += logp

        return x, logdet, eps, logp_sum

    def reverse(self, x, eps=None, eps_std=None):
        logdet = 0
        if eps is None:
            eps = [None] * (self.levels - 1)
        # reversing the elements and the indices
        for i, current_block in reversed(list(enumerate(self.blocks))):
            # print(f"iter: {i}")
            if i < self.levels - 1:
                x = split_reverse(x, eps[i], eps_std, self.zeroconv[i])
                # print(f"after split_reverse\niter: {i}\nx: {x.shape}")
            
            x, logdet = current_block.reverse(x, logdet)
            # print(f"logdet: {logdet.shape}\nx: {x.shape}")

        # unsqueeze before returning, equivalent to reversing the first squeeze before the loops in forward
        x = squeeze(x, reverse=True)
        return x


class Block(nn.Module):
    def __init__(self, in_channels, depth):
        super(Block, self).__init__()

        channel_after_squeeze = in_channels * 4  # because we apply a squeeze at the start
        self.flows = nn.ModuleList([StepFlow(channel_after_squeeze) for _ in range(depth)])

    def forward(self, x):
        logdet = 0
        for flow in self.flows:
            x, logds = flow(x)
            logdet = logdet + logds
        return x, logdet

    def reverse(self, x, logdet=None):
        for flow in reversed(self.flows):
            x, logdet = flow.reverse(x, logdet)
        return x, logdet

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

class StepFlow(nn.Module):
    def __init__(self, in_channels):
        super(StepFlow, self).__init__()

        assert in_channels % 2 == 0
        self.actnorm = ActNorm(in_channels)
        self.inconv = InvConv(in_channels)
        self.affine_coupling = AffineCouplingLayer(in_channels)

    def forward(self, x):
        x, logdet1 = self.actnorm(x)
        x, logdet2 = self.inconv(x)
        x, logdet3 = self.affine_coupling(x)

        return x, logdet1 + logdet2 + logdet3

    def reverse(self, x, logdet=None):
        x, logdet = self.affine_coupling.reverse(x, logdet)
        x, logdet = self.inconv.reverse(x, logdet)
        x, logdet = self.actnorm.reverse(x, logdet)

        return x, logdet


def squeeze(x: torch.Tensor, factor=2, reverse=False):
    batch_size, channels, height, width = x.size()

    if not reverse:
        x = x.reshape([batch_size, channels, height //
                       factor, factor, width // factor, factor])

        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape([batch_size, channels * factor * factor,
                       height // factor, width // factor])

    else:
        x = x.reshape([batch_size, channels // (factor * factor),
                       factor, factor, height, width])
        x = x.permute(0, 1, 4, 2, 5, 3)
        x = x.reshape([batch_size, channels // (factor * factor),
                       height * factor, width * factor])

    return x


def split(x, zeroconv):
    # x_a, x_b = torch.split(x, x.shape[1] // 2, 1)
    x_a, x_b = x.chunk(2, 1)
    gd = split_prior(x_a, zeroconv)

    #logdet += gd.logp(x_b)
    eps = gd.get_eps(x_b)

    return x_a, eps, gd.logp(x_b)


def split_prior(x_a, zeroconv):
    h = zeroconv(x_a)

    # mean = h[:, 0::2, :, :]
    # logs = h[:, 1::2, :, :]
    mean, logs = h.chunk(2, 1)

    gd = gaussian_diag(mean, logs)
    return gd


def split_reverse(x, eps, eps_std, zeroconv):
    x_a = squeeze(x, reverse=True)
    gd = split_prior(x_a, zeroconv)

    if eps is not None:
        x_b = gd.sample2(eps)
    elif eps_std is not None:
        # eps_std will probably broadcast to perform element-wise multiplication
        x_b = gd.sample2(gd.eps * torch.reshape(eps_std, [-1, 1, 1, 1]))
    else:
        x_b = gd.sample
    x = torch.cat((x_a, x_b), dim=1)

    return x


def gaussian_diag(mean, logsd):
    class o(object):
        pass

    o.mean = mean
    o.logsd = logsd
    o.eps = torch.normal(mean)
    o.sample = mean + torch.exp(logsd) * o.eps  # sample with Gaussian
    o.sample2 = lambda eps: mean + torch.exp(logsd) * eps  # sample with normalized gaussian
    o.logps = lambda x: -0.5 * (np.log(2 * np.pi) + 2. * logsd + (x - mean) ** 2 / torch.exp(
        2. * logsd))  # compute gaussian distribution
    o.logp = lambda x: torch.sum(o.logps(x), dim=[1, 2, 3])  # (iid) summation of log distributions
    o.get_eps = lambda x: (x - mean) / torch.exp(logsd)  # normalising to zero mean and unit variance
    return o
