import torch
import torch.nn as nn
from torch.nn import functional as F
from

logAbs = lambda: x: torch.log(torch.abs(x))


class ActNorm(nn.Module):
    """
    Implementation of the Actnorm layer
    Bias and scale are initialized as mean and standard deviation of the first minibatch
    Bias and scale are trainable after the init

    Based on:
        > github.com/openai/glow
        > https://github.com/rosinality/glow-pytorch
    """

    def __init__(self, in_channels, scale=1., logdet=True):
        super(ActNorm, self).__init__()
        # trainable params (assume 4d tensor on creation, change if detected in initialize)
        self.bias = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))

        self.in_channels = in_channels
        self.scale_factor = scale
        # buffer to check initialization
        self.register_buffer('initialized', torch.tensor(1)
        self.logdet=logdet

    # data-dependent weight initialization
    # we want to start off with zero mean and unit variance on the initial minibatch
    def initialize(self, x):
        with torch.no_grad():
            shape = x.shape()
            if len(shape) == 4:
                mean= dim_mean(x.clone(), dims=[0, 2,3], keepDims = True)
                variance = dim_mean((x.clone() + bias) ** 2, dims =[0,2,3], keepDims=True)
                std_dev = (self.scale_factor / (variance.sqrt() + 1e-6))

            elif len(shape) == 2:
                mean = dim_mean(x.clone(), dims=[0], keepDims = True)
                variance = dim_mean((x.clone() + bias) ** 2, dims=[0], keepDims=True)
                std_dev = (self.scale_factor / (variance.sqrt() + 1e-6))
                #reshape bias and scale tensors
                self.bias = nn.Parameter(torch.zeros(1, self.in_channels))
                self.scale = nn.Parameter(torch.zeros(1, self.in_channels))

            self.bias.data.copy_(-mean.data)
            # addition to avoid divsion by zero
            self.scale.data.copy_(std_dev.data)
            
            self.initialized += 1

    def forward(self, x):
        shape = x.shape()
        if len(shape) == 2:
            height, width = shape
        elif len(shape) == 4:
            height = shape[2]
            width = shape[3]
        else:
            raise Exception(f"Incorrect amount of dimensions in input data: should be 2 or 4, is {len(shape)}")

        # initialize on first item in buffer
        if self.initialized.item() == 0:
            self.initialize(x)

        log_abs=logAbs(self.scale)

        log_det=height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * x + self.bias, log_det #the original does this in separate helper functions, we don't need that IMO
        else:
            return self.scale * x + self.bias

    def reverse(self, output):
        return (output - self.bias)/self.scale

def dim_mean(tensor, dims=None: list, keepDims=False):
    """
    Utility function for averaging along multiple dimensions
    Adaptation of tf.reduce_mean
    Returns a tensor of mean values
    """
    if dims is None:
        return tensor.mean()
    else:
        dims=sorted(dims)
        for dim in dims:
            tensor=tensor.mean(dim, keepDims=True)
        if not keepDims:
            for idx, dim in enumerate(dims):
                tensor.squeeze_(d-i)
        return tensor
