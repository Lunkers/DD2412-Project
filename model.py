import torch
import torch.nn as nn
from affinecoupling import AffineCouplingLayer
from actnorm import ActNorm
from invconv import InvConv

class Glow(nn.Module):
    # multiple design choices here
    # one thing is whether we want to include the looping in or outside the model?
    # we also want to save the z's and logdets that are produced in the model I guess

    def __init__(self, x, levels, depth, if_logdet=True):
        super(Glow, self).__init__()
        # expecting shape of (batch_size, channels, height, width)
        # in_channels = x.shape[1]
        # assert in_channels % 2 == 0
        #
        # self.channels = in_channels
        self.z = nn.Parameter(torch.zeros(1))
        self.logdet = nn.Parameter(torch.zeros(1))
        self.levels = levels
        self.depth = depth
        self.step_flow = StepFlow(x, if_logdet)
        nn.ModuleList([self.step_flow])  # not sure if registered correctly

    def forward(self, z):

        for i in range(self.levels):
            for j in range(self.depth):
                z, logdet = self.step_flow(z)
            if i < self.levels-1:
                z, logdet = self.split(z)

    def squeeze(self, x: torch.Tensor, factor=2, reverse=False):
        batch_size, channels, height, width = x.size()

        if not reverse:
            x = x.reshape([batch_size, channels, height //
                           factor, factor, width // factor, factor])

            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape([batch_size, channels*factor*factor,
                           height//factor, width//factor])

        else:
            x = x.reshape([batch_size, channels // (factor * factor),
                          factor, factor, height, width])
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.reshape([batch_size, channels // (factor * factor),
                       height * factor, width * factor])

        return x

    def split(self, x):
        # should we include gaussianizing like they do in the code? There's a more involved split in the original code
        x_a, x_b = torch.split(x, x.shape[1] // 2, 1)
        return x_a, x_b

class StepFlow(nn.Module):
    def __init__(self, x, logdet=True):
        super(StepFlow, self).__init__()
        assert len(x.shape) == 4

        in_channels = x.shape[1]
        assert in_channels % 2 == 0

        self.actnorm = ActNorm(in_channels, logdet=logdet)
        self.inconv = InvConv(in_channels, logdet)
        self.affine_coupling = AffineCouplingLayer(in_channels)

        nn.ModuleList([self.actnorm, self.inconv, self.affine_coupling])  # not sure if this is correct, will need to double check

    def forward(self, x):
        z, logdet = self.actnorm(x)
        z, logdet = self.inconv(z, logdet)
        z = self.affine_coupling(z)  # logds is calculated
        # TODO: if necessary can refactor this into forward to return logdet, making it consistent with actnorm and inconv
        logdet = self.affine_coupling.logds(forward=True)

        return z, logdet


if __name__ == "__main__":
    # small test of squeeze

    x = torch.rand([1, 3, 32, 32])  # 32x32 RGB image
    print(x.size())
    y = squeeze(x)
    # y.size() should be [1, 12, 16, 16] for this test case
    print(y.size())
    z = squeeze(y, reverse=True)
    print(z.size())  # should be [1,3,32,32]
