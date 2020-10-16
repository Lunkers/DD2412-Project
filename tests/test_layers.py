import pytest
import torch
from actnorm import ActNorm
from invconv import InvConv
from affinecoupling import AffineCouplingLayer


# expected data shape (batch_size, channels, height, width)

def make_dummy_data(batch_size, channels, height, width):
    return torch.ones(batch_size, channels, height, width)


def make_default_dummy_data(batch_size_cifar, batch_size_mnist):
    dummy_cifar = torch.ones(batch_size_cifar, 3, 32, 32)
    dummy_mnist = torch.ones(batch_size_mnist, 1, 28, 28)

    return dummy_cifar, dummy_mnist


# default data
cifar, mnist = make_default_dummy_data(2, 2)
in_channel_cifar = cifar.shape[1]
in_channel_mnist = mnist.shape[1]


# TODO: Check that the right parameters are trained
# TODO: To check correction of forward this can be cumbersome because need to basically reimplement the essential operations, perhaps sufficient that we eyeball theoretically
class TestActnorm:
    """
        expected main behavior of actnorm:
        * output shape should be the same as input shape
        * logdet should be a single valued tensor
    """

    # omitting __init__ because pytest doesn't run classes with __init__ methods

    def forward_shape_output(self, in_channel, data):
        actnorm = ActNorm(in_channel)
        z, logdet = actnorm(data, if_logdet=True)

        assert z.shape == data.shape
        assert len(logdet.shape) == 0
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

        assert torch.all(z_reversed.eq(data.clone())).item()  # if the values are the same then the shapes are the same
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
        * output shape should be
        * logdet should be a single valued tensor
    """

    def forward_shape_output(self):
        return