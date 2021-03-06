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
    device = 'cpu'
    print(f"running on {device}")
    # set random seed for all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_set, test_set = get_dataloader(args.dataset, args.batch_size)
    x = next(iter(train_set))[0]  # extract first data from first batch

    net = Glow(in_channels=x.shape[1],
               depth=args.amt_flow_steps, levels=args.amt_levels, use_normalization=args.norm_method)
    net = net.to(device)

    assert os.path.isdir("checkpoints")
    checkpoint = torch.load(f"checkpoints/best_{args.dataset}.pth.tar")
    net.load_state_dict(checkpoint["model"])

    num_samples = args.n_samples
    sample_images = generate(net, num_samples, device, shape=x.shape, levels=args.amt_levels)

    os.makedirs('final_generation_img', exist_ok=True)
    grid = torchvision.utils.make_grid(sample_images, nrow=int(num_samples ** 0.5))
    # torchvision.utils.save_image(grid, f"generated_imgs/epoch_{epoch}.png")
    torchvision.utils.save_image(grid, f"final_generation_img/epoch_{100}_{args.dataset.lower()}.png", normalize=True, nrow=10,
                                 range=(-0.5, 0.5))

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
    channels, height, width = shape[1], shape[2], shape[3]
    x_shapes = create_x_shapes(channels, height, width, levels)
    temperature = 0.7
    x_sample = []
    for ch, h, w in x_shapes:
        x_random = torch.randn(n_samples, ch, h, w) * temperature
        x_sample.append(x_random.to(device))
    x = model.reverse(x_sample)
    #x /= 0.6  # attempt to make it brighter
    return x

def create_x_shapes(channels, height, width, levels):
    x_shapes = []
    for i in range(levels - 1):
        channels *= 2
        height //= 2
        width //= 2
        x_shapes.append((channels, height, width))

    x_shapes.append((channels * 4, height // 2, width // 2))
    return x_shapes

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
    parser.add_argument('--n_samples', '-S', default=64, type=int, help="samples to generate")
    parser.add_argument('--norm_method', default="", help="samples to generate")
    parser.add_argument('--dataset', default=DatasetEnum.CIFAR,
                        type=str, choices=[dataset.name for dataset in DatasetEnum])


    main(parser.parse_args())