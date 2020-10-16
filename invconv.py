import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class InvConv(nn.Module):
    """
    Invertible 1x1 convolution layer
    adapted from:
        > github.com/openAI/glow
    """

    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels
        # random initialization of convolution weights
        w_initialized = np.random.randn(num_channels, num_channels)
        # make orthogonal
        w_initialized = np.linalg.qr(w_initialized)[0].astype(np.float32)

        self.w = nn.Parameter(torch.from_numpy(w_initialized))

    def forward(self, x, logdet):
        shape = x.size()
        height, width = shape[2], shape[3]
        dlogdet = torch.slogdet(self.w)[1] * height * width
        _w = self.w.unsqueeze(2).unsqueeze(3)  # make weight matrix 4 dimensional for compatibility
        y = F.conv2d(x, _w)
        return y, logdet + dlogdet

    def reverse(self, y, logdet):
        shape = y.size()
        height, width = shape[2], shape[3]
        weight = torch.inverse(self.w)
        weight = weight.unsqueeze(2).unsqueeze(3)  # make weight matrix 4 dimensional for compatibility
        dlogdet = torch.slogdet(self.w)[1] * height * width
        x = F.conv2d(y, weight)
        return x, logdet - dlogdet
