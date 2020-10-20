import argparse
import numpy as np
import os
import random
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from preprocessing import Dataloader
from loss_utils import FlowNLL, bits_per_dimension
from averagemeter import AverageMeter
from model import Glow
from enum import Enum


def channels_from_dataset(dataset):
    if dataset == "MNIST":
        return 1
    else:
        return 3


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

    global best_loss
    if(args.generate_samples):
        print("generating samples")
    # load data
    # example for CIFAR-10 training:
    train_set, test_set = get_dataloader(args.dataset, args.batch_size)

    input_channels = channels_from_dataset(args.dataset)
    print(f"amount of  input channels: {input_channels}")
    # instantiate model
    # baby network to make sure training script works
    net = Glow(in_channels=input_channels,
               depth=args.amt_flow_steps, levels=args.amt_levels)

    net = net.to(device)

    start_epoch = 0
    # TODO: add functionality for loading checkpoints here
    if args.resume:
        print("resuming from checkpoint found in checkpoints/best.pth.tar.")
        # raise error if no checkpoint directory is found
        assert os.path.isdir("checkpoints")
        checkpoint = torch.load("checkpoints/best.pth.tar")
        net.load_state_dict(checkpoint["model"])
        global best_loss
        best_loss = checkpoint["test_loss"]
        start_epoch = checkpoint["epoch"]

    loss_function = FlowNLL().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # scheduler found in code, no mention in paper
    scheduler = sched.LambdaLR(
        optimizer, lambda s: min(1., s / args.warmup_iters))

    # should we add a resume function here?

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        print(f"training epoch {epoch}")
        train(net, train_set, device, optimizer, loss_function, scheduler)
        # how often do we want to test?
        if (epoch % 1== 0):  # revert this to 10 once we know that this works
            print(f"testing epoch {epoch}")
            test(net, test_set, device, loss_function, epoch, args.generate_samples)


@torch.enable_grad()
def train(model, trainloader, device, optimizer, loss_function, scheduler):
    """
    Trains the model for one epoch.
    Args:
        model: the model to train
        trainloader: Dataloader for the training data
        device: the compute device
        optimizer
        loss_function
    """
    global glbl_step
    model.train()
    loss_meter = AverageMeter("train-avg")
    for x, y in trainloader:
        x = x.to(device)
        optimizer.zero_grad()
        z, logdet, eps = model(x)
        # need to check how they formulate their loss function
        loss = loss_function(z, logdet)
        loss_meter.update(loss.item(), x.size(0))
        loss.backward()
        optimizer.step()
        scheduler.step(glbl_step)
        glbl_step += x.size(0)


@torch.no_grad()
def test(model, testloader, device, loss_function, epoch, generate_imgs):
    global best_loss  # keep track of best loss
    model.eval()
    loss = 0
    num_samples = 32  # should probably be an argument
    # TODO: add average loss checker here, they use that in code for checkpointing
    loss_meter = AverageMeter('test-avg')
    for x, y in testloader:
        x = x.to(device)
        z, logdet, _ = model(x)
        loss = loss_function(x, logdet)
        loss_meter.update(loss.item(), x.size(0))

    if loss_meter.avg < best_loss:
        print(f"New best model found, average loss {loss_meter.avg}")
        checkpoint_state = {
            "model": model.state_dict(),
            "test_loss": loss_meter.avg,
            "epoch": epoch
        }
        os.makedirs("checkpoints", exist_ok=True)
        # save the model
        torch.save(checkpoint_state, "checkpoints/best.pth.tar")
        best_loss = loss_meter.avg
    print(f"test epoch complete, result: {bits_per_dimension(x, loss_meter.avg)}")
    # generate samples after each test (?)
    if(generate_imgs):
        sample_images = generate(model, num_samples, device)
        os.makedirs('generated_imgs', exist_ok=True)
        grid = torchvision.utils.make_grid(sample_images, nrow=num_samples ** 0.5)
        torchvision.utils.save_image(grid, f"generated_imgs/epoch_{epoch}")


@torch.no_grad()
def generate(model, n_samples, device, n_channels=3):
    """
    Generate samples from the model
    args:
        model: the network model
        n_samples: amount of samples to generate
        device: the device we run the model on
        n_channels: the amount of channels for the output (usually 3, but on MNIST it's 1)
    """
    z = torch.randn((n_samples, n_channels, 32, 32),
                    dtype=torch.float32, device=device)
    # not sure how you guys implemented the reverse functionality
    x, _ = model.reverse(z)
    return x


class DatasetEnum(Enum):
    CIFAR = "CIFAR"
    MNIST = "MNIST"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DD2412 Mini-glow")
    # using CIFAR optimizations as default here
    parser.add_argument('--batch_size', default=512,
                        type=int, help="minibatch size")
    parser.add_argument('--lr', default=0.001, help="learning rate")
    parser.add_argument('--amt_channels', '-C', default=512,
                        help="amount of channels in the hidden layers")  # maybe remove this part? Don't have any way to pass it down atm
    parser.add_argument('--amt_levels', '-L', default=3, type=int,
                        help="amount of flow layers")
    parser.add_argument('--amt_flow_steps', '-K', type=int,
                        default=32, help="amount of flow steps")
    parser.add_argument('--seed', default=0, help="random seed")
    # no mention of this in the paper, but quite a lot in the code
    parser.add_argument('--warmup_iters', default=5000,
                        help="amount of iterations for learning rate warmup")
    parser.add_argument('--num_epochs', default=100, type=int,
                        help="number of epochs to train for")
    parser.add_argument('--resume', default=False,
                        help="Resume training of a saved model")
    parser.add_argument('--dataset', default=DatasetEnum.CIFAR,
                        type=str, choices=[dataset.name for dataset in DatasetEnum])
    parser.add_argument('--generate_samples', action="store_true", default=False)

    best_loss = 9999
    glbl_step = 0

    main(parser.parse_args())
