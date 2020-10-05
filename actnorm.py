import torch
import torch.nn as nn
from torch.nn import functional as F

logAbs = lambda: x: torch.log(torch.abs(x))

class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
      super().__init__()
      self.bias = nn.Parameter(torch.zeros(1, in_channel, 1,1))
      self.scale = nn.Parameter(torch.ones(1,in_channel, 1, 1))

      self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))
      self.logdet = logdet

    # data-dependent weight initialization
    # we want to start off with zero mean and unit variance on the initial minibatch
    def initialize(self, input):
        with torch.no_grad():
            flattened = input.permute(1,0,2,3).contiguous().view(input.shape[1], -1)
            mean = (
                flattened.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1,0,2,3)
            )
            std = (
                flattened.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1,0,2,3)
            )

            self.bias.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6)) #addition to avoid divsion by zero

    def forward(self, input):
        _, _, height, width = input.shape

        # initialize on first item in buffer
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill(1)

        log_abs = logAbs(self.scale)

        log_det = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * input + self.bias, log_det
        else:
            return self.scale * input + self.bias

    def reverse(self, output):
        return (output - self.bias)/self.scale
