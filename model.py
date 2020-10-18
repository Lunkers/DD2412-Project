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
    # multiple design choices here
    # one thing is whether we want to include the looping in or outside the model?
    # we also want to save the z's and logdets that are produced in the model I guess

    def __init__(self, x, levels, depth):
        super(Glow, self).__init__()
        # expecting shape of (batch_size, channels, height, width)
        # in_channels = x.shape[1]
        # assert in_channels % 2 == 0
        #
        # self.channels = in_channels
        self.z = nn.Parameter(torch.zeros(1))
        self.logdet = nn.Parameter(torch.zeros(1))
        self.levels = levels
        self.depth = depth
        self.step_flow = StepFlow(x)
        nn.ModuleList([self.step_flow])  # not sure if registered correctly

    def forward(self, x):
        logdet = 0
        eps = []
        x = squeeze(x)
        for i in range(self.levels):
            for j in range(self.depth):
                x, logdet = self.step_flow(x)
            if i < self.levels - 1:
                x, logdet, _eps = split(x, logdet)
                eps.append(_eps)
        return x, logdet, eps

    def reverse(self, x, eps=[], eps_std=None):
        for i in reversed(range(self.levels)):
            if i < self.levels - 1:
                x = split_reverse(x, eps, eps_std)
            for j in range(self.depth):
                x, logdet = self.step_flow.reverse(x, 0)

        return x


class StepFlow(nn.Module):
    def __init__(self, x):
        super(StepFlow, self).__init__()
        assert len(x.shape) == 4

        in_channels = x.shape[1]
        assert in_channels % 2 == 0

        self.actnorm = ActNorm(in_channels)
        self.inconv = InvConv(in_channels)
        self.affine_coupling = AffineCouplingLayer(in_channels)

        nn.ModuleList(
            [self.actnorm, self.inconv, self.affine_coupling])  # not sure if this is correct, will need to double check

    def forward(self, z, logdet=None):
        z, logdet = self.actnorm(z, logdet)
        z, logdet = self.inconv(z, logdet)
        z, logdet = self.affine_coupling(z, logdet)

        return z, logdet

    def reverse(self, z, logdet=None):
        z, logdet = self.affine_coupling.reverse(z, logdet)
        z, logdet = self.inconv.reverse(z, logdet)
        z, logdet = self.actnorm.reverse(z, logdet)

        return z, logdet


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
