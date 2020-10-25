import torch
from model import Glow
import os
import argparse
import random
import numpy as np
import torchvision
from enum import Enum
from preprocessing import Dataloader

def get_dataloader(dataset, batch_size):
    dataloader = Dataloader()
    cifar_train, cifar_test, mnist_train, mnist_test = dataloader.load_data(
        batch_size_cifar=batch_size, batch_size_mnist=batch_size)
    if dataset == "MNIST":
        return mnist_train, mnist_test
    if dataset == "CIFAR":
        return cifar_train, cifar_test

def main(args):
    # we're probably only be using 1 GPU, so this should be fine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"running on {device}")
    # set random seed for all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_set, test_set = get_dataloader(args.dataset, args.batch_size)
    x = next(iter(train_set))[0]  # extract first data from first batch

    net = Glow(in_channels=x.shape[1],
               depth=args.amt_flow_steps, levels=args.amt_levels)
    net = net.to(device)

    assert os.path.isdir("checkpoints")
    checkpoint = torch.load("checkpoints/best.pth.tar")
    net.load_state_dict(checkpoint["model"])

    num_samples = args.batch_size
    sample_images = generate(net, num_samples, device, shape=x.shape, levels=args.amt_levels)
    os.makedirs('final_generation_img', exist_ok=True)
    grid = torchvision.utils.make_grid(sample_images, nrow=int(num_samples ** 0.5))
    # torchvision.utils.save_image(grid, f"generated_imgs/epoch_{epoch}.png")
    torchvision.utils.save_image(grid, f"final_generation_img/epoch_{100}.png")

@torch.no_grad()
def generate(model, n_samples, device, shape, levels):
    """
    Generate samples from the model
    args:
        model: the network model
        n_samples: amount of samples to generate
        device: the device we run the model on
        n_channels: the amount of channels for the output (usually 3, but on MNIST it's 1)
    """
    # z = torch.randn((n_samples, n_channels, 32, 32),
    #                 dtype=torch.float32, device=device)
    channels, height, width = shape[1], shape[2], shape[3]
    # initial channels * 4 * 2^(levels-1) with L=4 and initial_channels=3 for cifar
    channels = int(channels * 4 * 2 ** (levels - 1))
    height = int(height // (2**levels))
    width = int(width // (2**levels))
    z = torch.randn((n_samples, channels, height, width), dtype=torch.float32, device=device)
    x = model.reverse(z)

    return x

class DatasetEnum(Enum):
    CIFAR = "CIFAR"
    MNIST = "MNIST"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DD2412 Mini-glow")
    # using CIFAR optimizations as default here
    parser.add_argument('--batch_size', default=16,
                        type=int, help="minibatch size")
    parser.add_argument('--amt_levels', '-L', default=4, type=int,
                        help="amount of flow layers")
    parser.add_argument('--amt_flow_steps', '-K', type=int,
                        default=8, help="amount of flow steps")
    parser.add_argument('--seed', default=0, help="random seed")
    parser.add_argument('--dataset', default=DatasetEnum.CIFAR,
                        type=str, choices=[dataset.name for dataset in DatasetEnum])


    main(parser.parse_args())