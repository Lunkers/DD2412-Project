import torch
import torch.nn as nn
import torch.optim as optim
from torch import optim as optim

from actnorm import ActNorm
from invconv import InvConv
from affinecoupling import AffineCouplingLayer
import random
import model as model
import numpy as np
from nn import NN

# TODO: To check correction of forward this can be cumbersome because need to basically reimplement the essential operations, perhaps sufficient that we eyeball theoretically

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
eps = 1e-2

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


def NLL(z, logdet, k=256):
    prior_log_likelihood = 0.5 * (z ** 2 + np.log(2 * np.pi))
    prior_log_likelihood = prior_log_likelihood.flatten(1).sum(-1) - np.log(k) * np.prod(z.size()[:1])
    log_likelihood = prior_log_likelihood + logdet
    negative_log_likelihood = -log_likelihood.mean()

    return negative_log_likelihood


def dummy_loss(z, logdet):
    return torch.sum(z) + torch.sum(logdet)


def train_step(data, logdet, net):
    optimizer = optim.Adam(net.parameters())
    optimizer.zero_grad()
    z, logdet = net(data, logdet)
    loss = NLL(z, logdet)
    loss.backward()
    optimizer.step()


def check_trainable_parameters(net, data, logdet=None, affine=False):
    trained = True

    before = []
    for param in net.parameters():
        # param data are somehow by reference, so need to clone
        before.append(param.data.clone())
        assert not (torch.all(param.eq(torch.zeros(param.shape))).item() and
                    torch.all(param.eq(torch.ones(param.shape))).item())

    if affine:
        for d in data:
            train_step(d, logdet, net)
    else:
        train_step(data, logdet, net)

    after = []
    for param in net.parameters():
        after.append(param.data)

    # expects every trainable parameter to be trained
    number_of_not_trained = 0
    iter = 0
    for be, af in zip(before, after):
        if torch.all(be.eq(af)).item():
            number_of_not_trained += 1
            print(f"iter: {iter}")
            trained = False
        iter += 1
    print(f"number_of_not_trained:{number_of_not_trained}, net_params: {len(list(net.parameters()))}")
    assert trained == True


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
        assert len(logdet.shape) == 0  # single valued tensor
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
        assert logdet_reversed.item() == 0  # single valued tensor
        assert list(actnorm.scale.shape) == [1, in_channel, 1, 1]
        assert list(actnorm.bias.shape) == [1, in_channel, 1, 1]

    def test_shape_and_value_reverse_cifar(self):
        self.reverse_shape_and_value(in_channel_cifar, cifar.clone())

    def test_shape_and_value_reverse_mnist(self):
        self.reverse_shape_and_value(in_channel_mnist, mnist.clone())

    def test_trainable_parameters_cifar(self):
        check_trainable_parameters(ActNorm(3), data=make_dummy_data(2, 3, 32, 32, random=True))

    def test_trainable_parameters_mnist(self):
        check_trainable_parameters(ActNorm(1), data=make_dummy_data(2, 1, 28, 28, random=True))


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
        dummy_logdet = make_dummy_logdet(random=True)
        z, logdet = invconv(data, dummy_logdet)
        z_reversed, logdet_reversed = invconv.reverse(z, logdet)

        assert len(z_reversed.shape) == len(data.shape)
        assert compute_max_relative_error(z_reversed, data) < eps
        assert logdet_reversed.item() == dummy_logdet
        assert list(invconv.w.shape) == [in_channel, in_channel]

    def test_reverse_shape_and_value_cifar(self):
        self.reverse_shape_and_value(3, make_dummy_data(2, 3, 32, 32, random=True))

    def test_reverse_shape_and_value_mnist(self):
        self.reverse_shape_and_value(1, make_dummy_data(2, 1, 28, 28, random=True))

    def test_trainable_parameters_cifar(self):
        check_trainable_parameters(InvConv(3), data=make_dummy_data(2, 3, 32, 32, random=True),
                                   logdet=make_dummy_logdet(random=True))

    def test_trainable_parameters_mnist(self):
        check_trainable_parameters(InvConv(1), data=make_dummy_data(2, 1, 28, 28, random=True),
                                   logdet=make_dummy_logdet(random=True))


class TestAffineCoupling:
    """
        expected main behavior of AffineCoupling:
        - output shape the same as input shape
        - logdet should be a 1D array consisting of sums of log dets for each datapoint in a given batch
    """

    def forward_shape_output(self, in_channel, data):
        affinecl = AffineCouplingLayer(in_channel)
        logdet = make_dummy_logdet(random=True)
        z, logdet = affinecl(data, logdet)
        height, width = data.shape[2], data.shape[3]

        assert list(z.shape) == [2, in_channel, height, width]
        assert len(logdet.shape) == 1  # batches of sums of log dets, so for each batch we have a sum of log dets

    def test_shape_output_cifar(self):
        self.forward_shape_output(4, make_dummy_data(2, 4, 32, 32, random=True))

    def test_shape_output_mnist(self):
        self.forward_shape_output(2, make_dummy_data(2, 2, 28, 28, random=True))

    def reverse_shape_and_value(self, in_channel, data):
        affinecl = AffineCouplingLayer(in_channel)
        dummy_logdet = make_dummy_logdet(random=True)
        z, logdet = affinecl(data, dummy_logdet)
        z_reversed, logdet_reversed = affinecl.reverse(z, logdet)

        assert len(z_reversed.shape) == len(data.shape)
        assert compute_max_relative_error(z_reversed, data) < eps
        # this is to make the tensor dimensionality compatible, the values of dummy_logdet stays the same
        dummy_logdet_with_batches = torch.empty(data.shape[0]).fill_(dummy_logdet)
        assert compute_max_relative_error(logdet_reversed, dummy_logdet_with_batches) < eps

    def test_reverse_shape_and_value_cifar(self):
        self.reverse_shape_and_value(4, make_dummy_data(2, 4, 32, 32, random=True))

    def test_reverse_shape_and_value_mnist(self):
        self.reverse_shape_and_value(2, make_dummy_data(2, 2, 28, 28, random=True))

    def test_conv_in_computational_graph(self):
        nn = NN(3)
        for param in nn.parameters():
            assert param.requires_grad

        afl = AffineCouplingLayer(3)
        for param in afl.parameters():
            assert param.requires_grad

    def test_trainable_parameters_cifar(self):
        data = []
        data.append(make_dummy_data(2, 12, 16, 16, random=True))
        data.append(make_dummy_data(2, 12, 16, 16, random=True))
        check_trainable_parameters(AffineCouplingLayer(12), data, make_dummy_logdet(random=True), affine=True)

    def test_trainable_parameters_mnist(self):
        data = []
        data.append(make_dummy_data(2, 4, 14, 14, random=True))
        data.append(make_dummy_data(2, 4, 14, 14, random=True))
        check_trainable_parameters(AffineCouplingLayer(4), data, make_dummy_logdet(random=True), affine=True)


"""
    Behavior of forward squeeze
    channel: channel*factor*factor
    height: height/2
    width: width/2
    
    Behavior for reverse is mirror of above
"""


def test_squeeze_shape():
    # cifar
    x = make_dummy_data(2, 3, 32, 32, random=True)
    y = model.squeeze(x)
    assert list(y.shape) == [2, 12, 16, 16]
    z = model.squeeze(y, reverse=True)
    assert list(z.shape) == [2, 3, 32, 32]

    # mnist
    x = make_dummy_data(2, 1, 28, 28, random=True)
    y = model.squeeze(x)
    assert list(y.shape) == [2, 4, 14, 14]
    z = model.squeeze(y, reverse=True)
    assert list(z.shape) == [2, 1, 28, 28]


"""
    Behavior of forward split
    z -> [batch_size, channels, height, width] = [batch_size, channels*2, height/2, width/2]
    logdet -> 1D tensor, remains same shape
    eps -> [batch_size, channels, height, width] = [batch_size, channels/2, height, width]
    
    Behavior of reverse split
    z -> original shape
"""


def split_shape(in_channel, data):
    batch_size, height, width = data.shape[0], data.shape[2], data.shape[3]
    # make logdet same shape as after affine coupling: 1D tensor with same number of entries as batch size
    dummy_logdet = torch.empty(data.shape[0]).fill_(make_dummy_logdet(random=True))
    z, logdet, eps = model.split(data, dummy_logdet)

    assert list(z.shape) == [batch_size, in_channel * 2, height / 2, width / 2]
    assert logdet.shape == dummy_logdet.shape
    assert list(eps.shape) == [batch_size, in_channel / 2, height, width]

    # multiple paths
    # eps != None or eps_std != None or both are None
    eps_std = make_dummy_data(1, 1, 1, 1, random=True)  # should be 1 single value
    z_with_eps = model.split_reverse(z, eps, None)
    z_with_eps_std = model.split_reverse(z, None, eps_std)
    z_without_any_eps = model.split_reverse(z, None, None)
    assert list(z_with_eps.shape) == [batch_size, in_channel, height, width]
    assert list(z_with_eps_std.shape) == [batch_size, in_channel, height, width]
    assert list(z_without_any_eps.shape) == [batch_size, in_channel, height, width]


# adjusted shape of cifar and mnist to after applying squeeze for testing robustness
def test_split_shape_cifar():
    split_shape(12, make_dummy_data(2, 12, 16, 16, random=True))


def test_split_shape_mnist():
    split_shape(4, make_dummy_data(2, 4, 14, 14, random=True))
