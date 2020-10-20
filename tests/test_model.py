import torch
import torch.nn as nn
import torch.optim as optim
from torch import optim as optim
import random
from model import Glow
import numpy as np
from preprocessing import Dataloader

manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
eps = 1e-2

cifar_train, cifar_test, mnist_train, mnist_test = Dataloader().load_data()


class TestGlow:

    def forward_and_reverse_output_shape(self, in_channel, data, levels=3, depth=4):
        glow = Glow(in_channel, levels, depth)
        z, logdet, eps = glow(data)
        height, width = data.shape[2], data.shape[3]

        """
            cifar example:
            Level = 3
            initial shape -> [4, 3, 32, 32]
            iter 1 -> z: [4, 12, 16, 16] because of squeeze from outside the loop
            iter 2 -> z: [4, 24, 8, 8] because of squeeze + split
            iter 3 -> z: [4, 48, 4, 4] because of squeeze + split
        """
        assert list(z.shape) == [4, in_channel * 4 * 2 ** (levels - 1), 4, 4]
        assert list(logdet.shape) == [4]  # because batch_size = 4
        assert len(eps) == levels - 1  # because L = 3 and split is executed whenever < L, i.e 2 times in total

        factor = 1
        for e in eps:
            factor *= 2
            # example: first eps -> from iter 1 take z shape and divide channel by 2: [4, 12/2, 16, 16]
            assert list(e.shape) == [4, in_channel * factor, height / factor, width / factor]

        """
            In total depth * levels = 4 * 3 = 12, so we got 12 instances of actnorm, inconv and affinecoupling
            Actnorm = 2 trainable parameters
            Invconv = 1 trainable parameter
            Affinecoupling = 6 trainable parameters (got 3 conv layers, each layer has weight + bias, so for all layers combined we get 6 in total)
            Zeroconv = 4
            
            12 * (2+1+6) + 4= 112
        """
        assert len(list(glow.parameters())) == (levels * depth) * (2 + 1 + 6) + 4
        for param in glow.parameters():
            assert param.requires_grad

        # reverse
        z = glow.reverse(z, eps)

        assert list(z.shape) == [4, 3, 32, 32]

    def test_forward_and_reverse_output_shape_cifar(self):
        # extract first batch of size 4 -> [train, labels] -> [[4,3,32,32], [4]]
        data = next(iter(cifar_train))
        train_data = data[0]
        self.forward_and_reverse_output_shape(train_data.shape[1], train_data)

    # def test_forward_and_reverse_output_shape_mnist(self):
    #     # extract first batch of size 4 -> [train, labels] -> [[4,1,28,28], [4]]
    #     data = next(iter(mnist_train))
    #     train_data = data[0]
    #     print(len(list(train_data.shape)))
    #     print(list(train_data.shape))
    #     # assert 1==2
    #     self.forward_and_reverse_output_shape(train_data.shape[1], train_data)
