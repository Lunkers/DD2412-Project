import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.linalg as linalg


def logabs(x): return torch.log(torch.abs(x))


class InvConv(nn.Module):
    """
    Invertible 1x1 convolution layer using LU decomposition
    adapted from:
        > github.com/openAI/glow
    """

    def __init__(self, num_channels):
        super(InvConv, self).__init__()
        self.num_channels = num_channels
        # random initialization of convolution weights
        w_init = np.random.randn(num_channels, num_channels)

        # use numpy and scipy for linalg operations
        q, _ = linalg.qr(w_init)  # we only want the orthongonal matrix
        P, L, U = linalg.lu(q.astype(np.float32))
        s = np.diag(U)
        U = np.triu(U, 1)  # we want an upper triangular matrix
        U_mask = np.triu(np.ones_like(U), 1)
        L_mask = U_mask.T  # L is lower triangular, we can transpose the U mask

        P = torch.from_numpy(P)
        U = torch.from_numpy(U)
        s = torch.from_numpy(s)
        L = torch.from_numpy(L)

        # we want to save these in the state dict, but not train them
        # thus, we use buffers
        self.register_buffer("P_buffer", P)
        self.register_buffer("U_mask_buffer", torch.from_numpy(U_mask))
        self.register_buffer("L_mask_buffer", torch.from_numpy(L_mask))
        self.register_buffer("s_sign_buffer", torch.sign(s))
        self.register_buffer("L_eye_buffer", torch.eye(L_mask.shape[0]))

        # we want to learn the parameters that actually create the weight matrix
        # see eq. 10 in Glow paper
        self.L = nn.Parameter(L)
        self.s = nn.Parameter(logabs(s))
        self.U = nn.Parameter(U)

    def forward(self, x, logdet):
        shape = x.size()
        height, width = shape[2], shape[3]
        weight = self.calculate_weight()
        dlogdet = torch.sum(self.s) * width * height #eq. 11
        y = F.conv2d(x, weight.unsqueeze(2).unsqueeze(3))
        return y, logdet + dlogdet

    def reverse(self, y, logdet):
        shape = y.size()
        height, width = shape[2], shape[3]
        weight = self.calculate_weight()
        dlogdet =  torch.sum(self.s) * width * height
        x = F.conv2d(y, weight.inverse().unsquueze(2).unsqueeze(3))
        return x, logdet - dlogdet

    def calculate_weight(self):
        """
        calculate Weight matrix W using LU decomposition
        """
        weight = self.P_buffer @ (self.L * self.L_mask_buffer + self.L_eye_buffer) @ (
            (self.U * self.U_mask_buffer) + torch.diag(self.s_sign_buffer * torch.exp(self.s)))

        return weight