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

        """
            Level = 3
            iter 1 -> z: [4, 12, 16, 16] because of squeeze from outside the loop
            iter 2 -> z: [4, 24, 8, 8] because of squeeze + split
            iter 3 -> z: [4, 48, 4, 4] because of squeeze + split
        """
        assert list(z.shape) == [4, 48, 4, 4]
        assert list(logdet.shape) == [4]  # because batch_size = 4
        assert len(eps) == 2  # because L = 3 and split is executed whenever < L, i.e 2 times in total
        assert list(eps[0].shape) == [4, 6, 16, 16]  # from iter 1 take z shape and divide channel by 2: [4, 12/2, 16, 16]
        assert list(eps[1].shape) == [4, 12, 8, 8]  # from iter 2 take z shape and divide channel by 2: [4, 24/2, 8, 8]

        """
            In total depth * levels = 4 * 3 = 12, so we got 12 instances of actnorm, inconv and affinecoupling
            Actnorm = 2 trainable parameters
            Invconv = 1 trainable parameter
            Affinecoupling = 6 trainable parameters (got 3 conv layers, each layer has weight + bias, so for all layers combined we get 6 in total)
            
            12 * 2 + 1 * 12 + 6 * 12 = 108
        """
        assert len(list(glow.parameters())) == (levels * depth) * (2 + 1 + 6)
        for param in glow.parameters():
            assert param.requires_grad

        # reverse
        z = glow.reverse(z, eps[-1])

        assert list(z.shape) == [4, 12, 16, 16]

        # assert 1==2

    def test_forward_and_reverse_output_shape_cifar(self):
        # extract first batch of size 4 -> [train, labels] -> [[4,3,32,32], [4]]
        data = next(iter(cifar_train))
        train_data = data[0]
        self.forward_and_reverse_output_shape(train_data.shape[1], train_data)
