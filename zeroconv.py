import torch
import torch.nn as nn


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channels, padding=0):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size=3, padding=padding)
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        # zero initialization
        torch.nn.init.zeros_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        return x * torch.exp(self.scale * 3)
