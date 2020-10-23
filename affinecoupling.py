import torch
import torch.nn as nn
from nn import NN as NN_func


class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels=512, additive_coupling=False):
        super(AffineCouplingLayer, self).__init__()
        self.NN = NN_func(in_channels, out_channels)

        self.scale_forward = torch.zeros(1)
        self.scale_reverse = torch.zeros(1)

        self.additive_coupling = additive_coupling

    def forward(self, x, input_logdet):
        # assuming shape is (batch_size, channels, height, width)
        x_a, x_b = x.chunk(2, dim=1)

        if self.additive_coupling:
            scale = 1
            y_a = x_a + self.NN(x_b)  # there is no scale when using additive coupling, only shift hence the adding
        else:
            h = self.NN(x_b)
            scale, shift = h[:, 0::2, ...], h[:, 1::2, ...]
            scale = torch.sigmoid(scale + 2.)
            y_a = scale *( x_a + shift)

        self.scale_forward = scale
        y_b = x_b
        y = torch.cat((y_a, y_b), dim=1)  # if channel is in the second dimension

        logdet = self.calc_logdet(reverse=False)
        input_logdet = input_logdet + logdet

        return y, input_logdet

    def reverse(self, y, input_logdet):
        # assuming shape is (batch_size, channels, height, width)
        y_a, y_b = torch.split(y, y.shape[1] // 2, 1)

        if self.additive_coupling:
            scale = 1
            x_a = y_a + self.NN(y_b)
        else:
            h = self.NN(y_b)
            scale, shift = torch.chunk(h, 2, 1)
            scale = torch.sigmoid(scale + 2.)
            x_a = (y_a - shift) / scale

        self.scale_reverse = scale
        x_b = y_b
        x = torch.cat((x_a, x_b), dim=1)  # if channel is in the second dimension

        logdet = self.calc_logdet(reverse=True)
        input_logdet = input_logdet - logdet

        return x, input_logdet

    # use logds only after forward or reverse has been calculated
    def calc_logdet(self, reverse=False):
        scale = self.scale_forward if not reverse else self.scale_reverse
        return torch.sum(torch.log(torch.abs(scale)), dim=[1, 2, 3])
