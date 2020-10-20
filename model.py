import torch
import torch.nn as nn
import numpy as np
from affinecoupling import AffineCouplingLayer
from actnorm import ActNorm
from invconv import InvConv


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
        self.initialize_blocks(in_channels)

    def initialize_blocks(self, in_channels):
        for _ in range(self.levels):
            self.blocks.append(Block(in_channels, self.depth))
            in_channels *= 2  # because we apply a split at the end of each level iteration and split has squeeze in it as well resulting in a factor of 2

    def forward(self, x):
        logdet = 0
        eps = []
        x = squeeze(x)
        for i, current_block in enumerate(self.blocks):
            # print(f"iter: {i}")
            x, logdet = current_block(x, logdet)
            # print(f"logdet: {logdet.shape}\nx: {x.shape}")
            if i < self.levels - 1:
                x, logdet, _eps = split(x, logdet)
                # print(f"after split\niter: {i}\nlogdet: {logdet.shape}\nx: {x.shape}")
                eps.append(_eps)
        return x, logdet, eps

    def reverse(self, x, eps=[], eps_std=None):
        logdet = 0
        for i, current_block in enumerate(reversed(self.blocks)): # a bit more readable, but might be an unnecessary change
            if i < self.levels - 1:
                x = split_reverse(x, eps, eps_std)
            x, logdet = current_block.reverse(x, logdet)

        return x


class Block(nn.Module):
    def __init__(self, in_channels, depth):
        super(Block, self).__init__()

        channel_after_squeeze = in_channels * 4  # because we apply a squeeze
        self.flows = nn.ModuleList([StepFlow(channel_after_squeeze) for _ in range(depth)])

    def forward(self, x, logdet=None):
        for flow in self.flows:
            x, logdet = flow(x, logdet)
        return x, logdet

    def reverse(self, x, logdet=None):
        # don't think logdet is relevant for reverse it seems in the original code, it's mostly ignored
        # We should still return it though, because the reverse() method in the model expects it
        for flow in reversed(self.flows):
            x, logdet = flow.reverse(x, logdet)
        return x, logdet


class StepFlow(nn.Module):
    def __init__(self, in_channels):
        super(StepFlow, self).__init__()

        assert in_channels % 2 == 0
        self.actnorm = ActNorm(in_channels)
        self.inconv = InvConv(in_channels)
        self.affine_coupling = AffineCouplingLayer(in_channels)

    def forward(self, x, logdet=None):
        x, logdet = self.actnorm(x, logdet)
        x, logdet = self.inconv(x, logdet)
        x, logdet = self.affine_coupling(x, logdet)

        return x, logdet

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


def split(x, logdet=0.):
    x_a, x_b = torch.split(x, x.shape[1] // 2, 1)
    gd = split_prior(x_a)

    logdet += gd.logp(x_b)
    x_a = squeeze(x_a)
    eps = gd.get_eps(x_b)

    return x_a, logdet, eps


def split_prior(x_a):
    in_channels = x_a.shape[1]
    out_channels = in_channels * 2
    h = zero_conv2D(x_a, in_channels, out_channels)

    mean = h[:, 0::2, :, :]
    logs = h[:, 1::2, :, :]

    gd = gaussian_diag(mean, logs)
    return gd


def split_reverse(x, eps, eps_std):
    x_a = squeeze(x, reverse=True)
    gd = split_prior(x_a)

    if eps is not None:
        x_b = gd.sample2(eps)
    elif eps_std is not None:
        # eps_std will probably broadcast to perform element-wise multiplication
        x_b = gd.sample2(gd.eps * torch.reshape(eps_std, [-1, 1, 1, 1]))
    else:
        x_b = gd.sample
    x = torch.cat((x_a, x_b), dim=1)

    return x


def zero_conv2D(x, in_channels, out_channels):
    zero_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    h = zero_conv(x)
    return h


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
