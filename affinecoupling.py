import torch
import torch.nn as nn
from nn import NN as NN_func

class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels, additive_coupling=False):
        super(AffineCouplingLayer, self).__init__()
        self.NN = NN_func(in_channels)

        self.s_forward = 0
        self.s_reverse = 0

        self.additive_coupling = additive_coupling

    def forward(self, x):
        # assuming shape is (batch_size, channels, a, b)
        x_a, x_b = torch.split(x, x.shape[1] // 2, 1)

        if self.additive_coupling:
            s = 1
            y_a = x_a # do they still add with t and use the NN function for additive coupling?
        else:
            s = self.NN(x_b)
            s,t = torch.split(s, s.shape[1] // 2, 1)
            s = torch.exp(torch.log(s))
            y_a = s * x_a + t #s and x_a has to be same shape in order to perform hadamard product

        self.s_forward = s
        y_b = x_b
        y = torch.cat((y_a, y_b), dim=1) # if channel is in the second dimension

        return y

    def reverse(self, y):
        # assuming shape is (batch_size, channels, a, b)
        y_a, y_b = torch.split(y, y.shape[1] // 2, 1)

        if self.additive_coupling:
            s = 1
            x_a = y_a # do they still add with t and use the NN function for additive coupling?
        else:
            s = self.NN(y_b)
            s,t = torch.split(s, s.shape[1] // 2, 1)
            s = torch.exp(torch.log(s))
            x_a = (y_a - t) / s #s and x_a has to be same shape in order to perform hadamard product

        self.s_reverse = s
        x_b = y_b
        x = torch.cat((x_a, x_b), dim=1) # if channel is in the second dimension

        return x

    def logds(self):
        return sum(torch.log(torch.abs(self.s_forward)))