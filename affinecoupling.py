import torch
import torch.nn as nn
from nn import NN as NN_func
from nn_with_normalization import NN as NN_norm
from zeroconv import ZeroConv2d
import torch.nn.functional as F


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels=512, additive_coupling=False):
        super(AffineCouplingLayer, self).__init__()
        self.NN = NN_func(in_channels, out_channels)
        # self.NN = NN_norm(in_channels, out_channels, use_normalization="batchnorm")
        self.additive_coupling = additive_coupling

    def forward(self, x):
        # assuming shape is (batch_size, channels, height, width)
        x_a, x_b = x.chunk(2, dim=1)

        if self.additive_coupling:
            scale = 1
            y_a = x_a + self.NN(x_b)  # there is no scale when using additive coupling, only shift hence the adding
        else:
            h = self.NN(x_a)
            # scale, shift = h[:, 0::2, ...], h[:, 1::2, ...]
            scale, shift = h.chunk(2, 1)
            scale = torch.sigmoid(scale + 2.)
            y_b = scale * (x_b + shift)

        y_a = x_a
        y = torch.cat([y_a, y_b], dim=1)  # if channel is in the second dimension

        logdet = self.calc_logdet(scale)

        return y, logdet

    def reverse(self, y):
        # assuming shape is (batch_size, channels, height, width)
        y_a, y_b = y.chunk(2, 1)

        if self.additive_coupling:
            scale = 1
            x_a = y_a + self.NN(y_b)
        else:
            h = self.NN(y_a)
            scale, shift = torch.chunk(h, 2, 1)
            scale = torch.sigmoid(scale + 2.)
            x_b = (y_b - shift) / scale  # however in ros they do y_b / scale first and then subtract shift

        x_a = y_a
        x = torch.cat([x_a, x_b], dim=1)  # if channel is in the second dimension

        # logdet = self.calc_logdet(scale)
        # input_logdet = input_logdet - logdet

        return x

    # use logds only after forward or reverse has been calculated
    def calc_logdet(self, scale):
        return torch.sum(torch.log(torch.abs(scale)), dim=[1, 2, 3])
