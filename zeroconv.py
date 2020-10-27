import torch.nn as nn
import torch
import torch.nn.functional as F

class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(ZeroConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channels, kernel_size=3, padding=0)
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

        torch.nn.init.zeros_(self.conv.weight)
        torch.nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = F.pad(x, [1, 1, 1, 1], value=1)
        return self.conv(x) * torch.exp(self.scale * 3)