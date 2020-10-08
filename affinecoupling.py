import torch
import torch.nn as nn
from nn import NN as NN_func


class AffineCouplingLayer(nn.Module):
    def __init__(self, z, additive_coupling=False):
        super(AffineCouplingLayer, self).__init__()
        assert len(z.shape) == 4  # expect data to be (batch_size, channels, height, width)
        in_channels, out_channels = z.shape[2], z.shape[3]
        self.NN = NN_func(in_channels, out_channels)

        self.scale_forward = torch.zeros(1)
        self.scale_reverse = torch.zeros(1)

        self.additive_coupling = additive_coupling

    def forward(self, x):
        # assuming shape is (batch_size, channels, height, width)
        x_a, x_b = torch.split(x, x.shape[1] // 2, 1)

        if self.additive_coupling:
            scale = 1
            y_a = x_a + self.NN(x_b)  # there is no scale when using additive coupling, only shift hence the adding
        else:
            h = self.NN(x_b)
            shift = h[:, 0::2, :, :]  # s = scale and t = shift
            scale = torch.sigmoid(torch.log(h[:, 1::2, :, :]))  # they add 2. in the code but i guess we can skip
            y_a = scale * x_a + shift

        self.s_forward = scale
        y_b = x_b
        y = torch.cat((y_a, y_b), dim=1)  # if channel is in the second dimension

        return y

    def reverse(self, y):
        # assuming shape is (batch_size, channels, height, width)
        y_a, y_b = torch.split(y, y.shape[1] // 2, 1)

        if self.additive_coupling:
            scale = 1
            x_a = y_a + self.NN(y_b)
        else:
            h = self.NN(y_b)
            shift = h[:, 0::2, :, :]
            scale = torch.sigmoid(torch.log(h[:, 1::2, :, :]))  # they add 2. in the code, but i guess we can skip
            x_a = (y_a - shift) / scale

        self.scale_reverse = scale
        x_b = y_b
        x = torch.cat((x_a, x_b), dim=1)  # if channel is in the second dimension

        return x

    # use logds only after forward or reverse has been calculated
    def logds(self, forward):
        scale = self.scale_forward if forward else self.scale_reverse
        return torch.sum(torch.log(torch.abs(scale)), dim=[1, 2, 3])
