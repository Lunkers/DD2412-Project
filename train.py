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


def main(args):
    # we're probably only be using 1 GPU, so this should be fine
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # set random seed for all
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    global best_loss
    dataloader = Dataloader()

    # load data
    # example for CIFAR-10 training:
    cifar_train, cifar_test, _, _ = dataloader.load_data(
        batch_size_cifar=args.batch_size, batch_size_mnist=args.batch_size)

    # instantiate model
    net = Glow(flow_steps=args.amt_flow_steps, layers=args.amt_flow_layers,
               channels=args.amt_channels)  # or whatever we name the class in the end

    net = net.to(device)

    start_epoch = 0
    # TODO: add functionality for loading checkpoints here
    if args.resume:
        print("resuming from checkpoint found in checkpoints/best.pth.tar.")
        assert os.path.isdir("checkpoints") #raise error if no checkpoint directory is found
        checkpoint = torch.load("checkpoints/best.pth.tar")
        net.load_state_dict(checkpoint["model"])
        global best_loss
        best_loss = checkpoint["test_loss"]
        start_epoch = checkpoint["epoch"]

    loss_function = FlowNLL()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    # scheduler found in code, no mention in paper
    scheduler = sched.LambdaLR(
        optimizer, lambda s: min(1., s / args.warmup_iters))

    # should we add a resume function here?

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        train(net, cifar_train, device, optimizer, loss_function, scheduler)
        # how often do we want to test?
        if (epoch % 10 == 0):
            test(model, cifar_test, device, loss_function, epoch)


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

    net.train()
    loss_meter = AverageMeter("train-avg")
    for x, y in trainloader:
        x.to(device)
        optimizer.zero_grad()
        z, logdet = model(x)
        # need to check how they formulate their loss function
        loss = loss_function(z, logdet)
        loss_meter.update(loss.item(), x.size(0))
        loss.backward()
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def test(model, testloader, device, loss_function, epoch):
    global best_loss  # keep track of best loss
    loss = 0
    num_samples = 32  # should probably be an argument
    # TODO: add average loss checker here, they use that in code for checkpointing
    loss_meter = AverageMeter('test-avg')
    for x, y in testloader:
        x = x.to(device)
        z, logdet = model(x)
        loss = loss_function(x, logdet)
        loss_meter.update(loss.item(), x.size(0))

    if loss_meter.avg < best_loss:
        print(f"New best model found, average loss {loss_meter.avg}")
        checkpoint_state = {
            "model": model.state_dict(),
            "test_loss": avg_loss,
            "epoch": epoch
        }
        os.makedirs("checkpoints", exist_ok=True)
        # save the model
        torch.save(checkpoint_state, "checkpoints/best.pth.tar")
        best_loss = loss_meter.avg
    print(f"test epoch complete, result: {bits_per_dimension(x, loss.avg)}")
    # generate samples after each test (?)
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
    x, _ model(z).reverse()
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DD2412 Mini-glow")
    # using CIFAR optimizations as default here
    parser.add_argument('--batch_size', default=512,
                        type=int, help="minibatch size")
    parser.add_argument('--lr', default=0.001, help="learning rate")
    parser.add_argument('--amt_channels', '-C', default=512,
                        help="amount of channels in the hidden layers")
    parser.add_argument('--amt_layers', '-L', default=3,
                        help="amount of flow layers")
    parser.add_argument('--amt_flow-steps', '-K',
                        default=32, help="amount of flow steps")
    parser.add_argument('--seed', default=0, help="random seed")
    # no mention of this in the paper, but quite a lot in the code
    parser.add_argument('--warmup_iters', default=5000,
                        help="amount of iterations for learning rate warmup")
    parser.add_argument('--num_epochs', default=100,
                        help="number of epochs to train for")
    parser.add_argument('--resume', default=False,
                        help="Resume training of a saved model")
    best_loss = 9999

    main(parser.parse_args())
