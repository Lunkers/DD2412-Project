import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, in_channels, out_channels=512, ker_size_3=3, ker_size_1=1):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels // 2, out_channels, ker_size_3, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, ker_size_1)
        self.conv3 = nn.Conv2d(out_channels, in_channels, ker_size_3)
        # zero initialization
        torch.nn.init.zeros_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)


    def forward(self, x):
        s = F.relu(self.conv1(x))
        s = F.relu(self.conv2(s))
        return self.conv3(s)