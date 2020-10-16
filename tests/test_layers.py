import pytest
import torch
from actnorm import ActNorm
from invconv import InvConv
from affinecoupling import AffineCouplingLayer
import random
import numpy as np

# TODO: Check that the right parameters are trained
# TODO: To check correction of forward this can be cumbersome because need to basically reimplement the essential operations, perhaps sufficient that we eyeball theoretically

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
eps = 1e-3

use_cuda = True
pin_memory = False
no_memory_block = False
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.cuda.manual_seed(manualSeed)
    pin_memory = True
    no_memory_block = True


# expected data shape (batch_size, channels, height, width)

def make_dummy_data(batch_size, channels, height, width, random=False):
    if random:
        return torch.randn(batch_size, channels, height, width)
    return torch.ones(batch_size, channels, height, width)


def make_default_dummy_dataset(batch_size_cifar, batch_size_mnist):
    dummy_cifar = torch.ones(batch_size_cifar, 3, 32, 32)
    dummy_mnist = torch.ones(batch_size_mnist, 1, 28, 28)

    return dummy_cifar, dummy_mnist


def make_dummy_logdet(random=False):
    if random:
        return torch.randn(1).item()
    return torch.tensor(1).item()


def compute_max_relative_error(x, y):
    return torch.max(torch.abs(x - y) /
                     torch.max(torch.empty(x.shape).fill_(eps), torch.abs(x) + torch.abs(y)))


# default data
cifar, mnist = make_default_dummy_dataset(2, 2)
in_channel_cifar = cifar.shape[1]
in_channel_mnist = mnist.shape[1]

class TestActnorm:
    """
        expected main behavior of actnorm:
        - output shape should be the same as input shape
        - logdet should be a single valued tensor
    """

    # omitting __init__ because pytest doesn't run classes with __init__ methods

    def forward_shape_output(self, in_channel, data):
        actnorm = ActNorm(in_channel)
        z, logdet = actnorm(data, if_logdet=True)

        assert z.shape == data.shape  # input shape and output shape should be same
        assert len(logdet.shape) == 0  # logdet only a single valued tensor
        # scale and bias parameters should only work on the channel entries of the input
        assert list(actnorm.scale.shape) == [1, in_channel, 1, 1]
        assert list(actnorm.bias.shape) == [1, in_channel, 1, 1]

    def test_shape_output_cifar(self):
        self.forward_shape_output(in_channel_cifar, cifar.clone())

    def test_shape_output_mnist(self):
        self.forward_shape_output(in_channel_mnist, mnist.clone())

    def reverse_shape_and_value(self, in_channel, data):
        actnorm = ActNorm(in_channel)
        z, logdet = actnorm(data, if_logdet=True)
        z_reversed, logdet_reversed = actnorm.reverse(z, logdet)

        assert torch.all(z_reversed.eq(data)).item()  # if the values are the same then the shapes are the same
        assert logdet_reversed.item() == 0
        assert list(actnorm.scale.shape) == [1, in_channel, 1, 1]
        assert list(actnorm.bias.shape) == [1, in_channel, 1, 1]

    def test_shape_and_value_reverse_cifar(self):
        self.reverse_shape_and_value(in_channel_cifar, cifar.clone())

    def test_shape_and_value_reverse_mnist(self):
        self.reverse_shape_and_value(in_channel_mnist, mnist.clone())


class TestInvConv:
    """
        expected main behavior of InvConv:
        - output shape should be same as input shape
        - logdet should be a single valued tensor
    """

    def forward_shape_output(self, in_channel, data):
        invconv = InvConv(in_channel)
        logdet = make_dummy_logdet(random=True)
        z, logdet = invconv(data, logdet)
        height, width = data.shape[2], data.shape[3]

        assert list(z.shape) == [2, in_channel, height, width]
        assert len(logdet.shape) == 0
        assert list(invconv.w.shape) == [in_channel, in_channel]

    def test_shape_output_cifar(self):
        # this should be equivalent to a modified cifar10 from actnorm
        self.forward_shape_output(3, make_dummy_data(2, 3, 32, 32, random=True))

    def test_shape_output_mnist(self):
        # this should be equivalent to a modified mnist from actnorm
        self.forward_shape_output(1, make_dummy_data(2, 1, 28, 28, random=True))

    def reverse_shape_and_value(self, in_channel, data):
        invconv = InvConv(in_channel)
        dummy_logdet = make_dummy_logdet()
        z, logdet = invconv(data, dummy_logdet)
        z_reversed, logdet_reversed = invconv.reverse(z, logdet)

        assert len(z_reversed.shape) == len(data.shape)
        assert compute_max_relative_error(z_reversed, data) < eps
        assert logdet_reversed.item() == dummy_logdet
        assert list(invconv.w.shape) == [in_channel, in_channel]

    def test_reverse_shape_and_value_cifar(self):
        self.reverse_shape_and_value(in_channel_cifar, make_dummy_data(2, 3, 32, 32, random=True))

    def test_reverse_shape_and_value_mnist(self):
        self.reverse_shape_and_value(in_channel_mnist, make_dummy_data(2, 1, 28, 28, random=True))

