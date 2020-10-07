import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineCouplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels=512, additive_coupling = False):
        super(AffineCouplingLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, 3, 3)
        # zero initialization
        torch.nn.init.zeros_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)

        # can change this to register_buffer, just need to figure out the dimens of x_a and y_a
        # but then need to first solve the issue with halving input channel, where it's RGB for cifar
        self.s_forward = 0
        self.s_reverse = 0

        self.additive_coupling = additive_coupling

    def forward(self, x):
        # assuming shape is (batch_size, channels, a, b)
        x_a, x_b = torch.split(x, x.shape[1]/2, 1) # the input channel is 3, so this will not be even, should I add a dimen with squeeze or pad it?

        if self.additive_coupling:
            s = 1
            y_a = x_a # do they still add with t and use the NN function for additive coupling?
        else:
            s = F.relu(self.conv1(x_b))
            s = F.relu(self.conv2(s))
            s = self.conv3(s)
            s,t = torch.split(s, s.shape[1]/2, 1)
            s = torch.exp(torch.log(s))
            y_a = s * x_a + t #s and x_a has to be same shape in order to perform hadamard product

        self.s_forward = s
        y_b = x_b
        y = torch.cat((y_a, y_b), dim=1) # if channel is in the second dimension

        return y

    def reverse(self, y):
        # assuming shape is (batch_size, channels, a, b)
        y_a, y_b = torch.split(y, y.shape[1]/2, 1) # the input channel is 3, so this will not be even, should I add a dimen with squeeze or pad it?

        if self.additive_coupling:
            s = 1
            x_a = y_a # do they still add with t and use the NN function for additive coupling?
        else:
            s = F.relu(self.conv1(y_b))
            s = F.relu(self.conv2(s))
            s = self.conv3(s)
            s,t = torch.split(s, s.shape[1]/2, 1)
            s = torch.exp(torch.log(s))
            x_a = s * y_a + t #s and x_a has to be same shape in order to perform hadamard product

        self.s_reverse = s
        x_b = y_b
        x = torch.cat((x_a, x_b), dim=1) # if channel is in the second dimension

        return x

    def logds(self):
        return sum(torch.log(torch.abs(self.s_forward)))