import torch
import torch.nn as nn
import torch.nn.functional as F
from actnorm import ActNorm
from zeroconv import ZeroConv2d


class NN(nn.Module):
    def __init__(self, in_channels, out_channels=512, ker_size_3=3, ker_size_1=1, use_normalization=""):
        super(NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels // 2, out_channels, ker_size_3, padding=2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, ker_size_1)
        self.conv3 = ZeroConv2d(out_channels, in_channels)
        self.use_normalization = use_normalization
        if use_normalization == "actnorm":
            norm = ActNorm
        elif use_normalization == "batchnorm":
            norm = nn.BatchNorm2d
        else:
            norm = None

        if norm:
        # Chose to do actnorm after each conv according to openai, chris used before each conv
            self.normalize1 = norm(out_channels)
            self.normalize2 = norm(out_channels)
            self.normalize3 = norm(in_channels)

        # manually initialize with normal dist instead of kaiming_uniform:
        # https://github.com/pytorch/pytorch/blob/ce5bca5502790e83ca8db25adcdd4694aa636c46/torch/nn/modules/conv.py#L111
        torch.nn.init.normal_(self.conv1.weight, 0, 0.05)
        torch.nn.init.zeros_(self.conv1.bias)

        torch.nn.init.normal_(self.conv2.weight, 0, 0.05)
        torch.nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        if self.use_normalization == "actnorm":
            x = F.relu(self.normalize1(self.conv1(x), use_logdet=False))
            x = F.relu(self.normalize2(self.conv2(x), use_logdet=False))
            return self.normalize3(self.conv3(x), use_logdet=False)
        elif self.use_normalization == "batchnorm":
            x = F.relu(self.normalize1(self.conv1(x)))
            x = F.relu(self.normalize2(self.conv2(x)))
            return self.normalize3(self.conv3(x))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            return self.conv3(x)
