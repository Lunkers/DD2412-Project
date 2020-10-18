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

use_cuda = True
pin_memory = False
no_memory_block = False
device = torch.device('cuda' if use_cuda else 'cpu')
if use_cuda:
    torch.cuda.manual_seed(manualSeed)
    pin_memory = True
    no_memory_block = True

cifar_train, cifar_test, mnist_train, mnist_test = Dataloader().load_data()

class TestGlow:

    def forward_output_shape(self, in_channel, data, levels=3, depth=4):
        glow = Glow(data, levels, depth)
        z, logdet, eps = glow(data)
        assert 1==2

    def test_forward_output_shape_cifar(self):
        # extract first batch of size 4 -> [train, labels] -> [[4,3,32,32], [4]]
        data = next(iter(cifar_train))
        train_data = data[0]
        self.forward_output_shape(train_data.shape[1], train_data)
