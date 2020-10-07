import torch
import torch.nn as nn
import torch.nn.functional as F

class NN(nn.Module):
    def __init__(self, in_channels, out_channels=512):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 1)
        self.conv3 = nn.Conv2d(out_channels, 3, 3)
        # zero initialization
        torch.nn.init.zeros_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)


    def forward(self, x):
        s = F.relu(self.conv1(x))
        s = F.relu(self.conv2(s))
        return self.conv3(s)
