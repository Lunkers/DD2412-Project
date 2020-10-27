import torch
import torch.nn as nn
import numpy as np
from affinecoupling import AffineCouplingLayer
from actnorm import ActNorm
from invconv import InvConv
from math import log, pi
from torch.nn import functional as F
from zeroconv import ZeroConv2d


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
        for _ in range(self.levels-1):
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
            if i < self.levels - 1:  # last block shouldnt split
                x, _eps, logp = split(x, self.zeroconv[i])
                eps.append(_eps)
            elif i == len(self.blocks) - 1:  # perform this at last iteration
                zero = torch.zeros_like(x)
                mean, log_sd = self.zeroconv[i](zero).chunk(2, 1)
                logp = gaussian_log_p(x, mean, log_sd)
                logp = logp.sum(dim=[1, 2, 3])
                eps.append(x)
            logp_sum += logp

        return x, logdet, eps, logp_sum

    def reverse(self, x):
        for i, current_block in reversed(list(enumerate(self.blocks))):
            if i < self.levels - 1:
                output = split_reverse(output, x[i], self.zeroconv[i])
            else:
                zero = torch.zeros_like(x[i])
                mean, log_sd = self.zeroconv[i](zero).chunk(2, 1)
                output = gaussian_sample(x[i], mean, log_sd)

            output = current_block.reverse(output)
            output = squeeze(output, reverse=True)
        return output

def split_reverse(x, eps, zeroconv):
    mean, log_sd = zeroconv(x).chunk(2, 1)
    z = gaussian_sample(eps, mean, log_sd)
    return torch.cat([x, z], dim=1)


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

    def reverse(self, x):
        for flow in reversed(self.flows):
            x = flow.reverse(x)
        return x

def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps

def normalize_data(x, mean, logsd):
    return (x - mean) / torch.exp(logsd)

def sample(mean, logsd):
    return mean + torch.exp(logsd) * torch.normal(mean)

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

    def reverse(self, x):
        x = self.affine_coupling.reverse(x)
        x = self.inconv.reverse(x)
        x = self.actnorm.reverse(x)

        return x


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

    eps = gd.get_eps(x_b)
    #eps = x_b  # rosalinty does not normalize it

    return x_a, eps, gd.logp(x_b)


def split_prior(x_a, zeroconv):
    h = zeroconv(x_a)

    # mean = h[:, 0::2, :, :]
    # logs = h[:, 1::2, :, :]
    mean, logs = h.chunk(2, 1)

    gd = gaussian_diag(mean, logs)
    return gd


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
