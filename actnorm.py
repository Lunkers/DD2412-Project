import torch
import torch.nn as nn
from torch.nn import functional as F


logAbs = lambda x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    """
    Implementation of the Actnorm layer
    Bias and scale are initialized as mean and standard deviation of the first minibatch
    Bias and scale are trainable after the init

    Based on:
        > github.com/openai/glow
        > https://github.com/rosinality/glow-pytorch
    """

    def __init__(self, in_channels, scale=1.):
        super(ActNorm, self).__init__()
        # trainable params (assume 4d tensor on creation, change if detected in initialize)
        self.bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        self.in_channels = in_channels
        self.scale_factor = scale
        # buffer to check initialization
        self.register_buffer('initialized', torch.tensor(0))

    # data-dependent weight initialization
    # we want to start off with zero mean and unit variance on the initial minibatch
    def initialize(self, x):
        with torch.no_grad():
            flatten = x.permute(1, 0, 2, 3).contiguous().view(x.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.bias.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))
            self.initialized.fill_(1)

    def forward(self, x, input_logdet=None, if_logdet=True):
        shape = x.shape
        height, width = self.get_shape(shape)

        # initialize on first item in buffer
        if self.initialized.item() == 0:
            print("Initializing actnorm")
            self.initialize(x)

        log_det = self.calc_logdet(height, width)
        if input_logdet != None:
            log_det = log_det + input_logdet


        if if_logdet:
            # the original does this in separate helper functions, we don't need that IMO
            return self.scale * x + self.bias, log_det
        else:
            return self.scale * x + self.bias

    def get_shape(self, shape):
        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 4:
            height = shape[2]
            width = shape[3]
        else:
            raise Exception(
                f"Incorrect amount of dimensions in input data: should be 2 or 4, is {len(shape)}")
        return height, width

    def calc_logdet(self, height, width):
        log_abs = logAbs(self.scale)
        log_det = height * width * torch.sum(log_abs)

        return log_det

    def reverse(self, output, input_logdet=None):
        height, width = self.get_shape(output.shape)
        output = (output - self.bias) / self.scale

        if input_logdet != None:
            input_logdet = input_logdet - self.calc_logdet(height, width)
            return output, input_logdet
        return output
