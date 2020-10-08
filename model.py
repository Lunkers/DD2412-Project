import torch


def squeeze(x: torch.Tensor, factor=2, reverse=False):
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


if __name__ == "__main__":
    # small test of squeeze

    x = torch.rand([1, 3, 32, 32])  # 32x32 RGB image
    print(x.size())
    y = squeeze(x)
    # y.size() should be [1, 12, 16, 16] for this test case
    print(y.size())
    z = squeeze(y, reverse=True)
    print(z.size())  # should be [1,3,32,32]
