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
from torch import autograd
import csv

def write_arr_to_csv(arr, filename):
    """
    outputs an array of dictionaries to a CSV file
    Args:
        arr: an array of dictionaries
        filename: name of the output file
    """
    keys = arr[0].keys()
    with open(f"{filename}.csv", "w", newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(arr)



def channels_from_dataset(dataset):
    if dataset == "MNIST":
        return 3 #openAI apparently stack MNIST to be 3 channels
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


# same preprocessing as rosalinty
def preprocess(x, n_bins=256):
    x = (x * 255) / n_bins - 0.5
    return x + torch.rand_like(x) / n_bins

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
    # # baby network to make sure training script works
    net = Glow(in_channels=input_channels,
               depth=args.amt_flow_steps, levels=args.amt_levels)

    # code for rosalinty model
    # net = RosGlow(input_channels, args.amt_flow_steps, args.amt_levels)

    net = net.to(device)

    print(f"training for {args.num_epochs} epochs.")

    start_epoch = 0
    # TODO: add functionality for loading checkpoints here
    if args.resume:
        print("resuming from checkpoint found in checkpoints/best_{args.dataset.lower()}.pth.tar.")
        # raise error if no checkpoint directory is found
        assert os.path.isdir("new_checkpoints")
        checkpoint = torch.load(f"new_checkpoints/best_{args.dataset.lower()}.pth.tar")
        net.load_state_dict(checkpoint["model"])
        global best_loss
        best_loss = checkpoint["test_loss"]
        start_epoch = checkpoint["epoch"]

    loss_function = FlowNLL().to(device)
    optimizer = optim.Adam(net.parameters(), lr=float(args.lr))
    # scheduler found in code, no mention in paper
    # scheduler = sched.LambdaLR(
    #     optimizer, lambda s: min(1., s / args.warmup_iters))

    # should we add a resume function here?

    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        print(f"training epoch {epoch}")
        train(net, train_set, device, optimizer, loss_function, epoch)
        # how often do we want to test?
        if (epoch % 10 == 0):  # revert this to 10 once we know that this works
            print(f"testing epoch {epoch}")
            test(net, test_set, device, loss_function, epoch, args.generate_samples, args.amt_levels, args.dataset)


@torch.enable_grad()
def train(model, trainloader, device, optimizer, loss_function, epoch):
    """
    Trains the model for one epoch.
    Args:
        model: the model to train
        trainloader: Dataloader for the training data
        device: the compute device
        optimizer
        loss_function
    """
    global train_losses
    model.train()
    train_iter = 0
    loss_meter = AverageMeter("train-avg")
    for x, _ in trainloader:
        x = x.to(device)
        z, logdet, _, logp = model(preprocess(x))
        loss = loss_function(logp, logdet, x.size())

        # code for rosalinty model
        # log_p_sum, logdet, z_outs = model(preprocess(x))
        # loss = loss_function(log_p_sum, logdet, x.size())

        if(train_iter % 10 == 0):
            print(f"iteration: {train_iter}, loss: {loss.item()}", end="\r")
        
        model.zero_grad()
        loss_meter.update(loss.item())
        loss.backward()
        optimizer.step()
        train_iter += 1
    print(f"epoch complete, mean loss: {loss_meter.avg}")
    train_losses.append({"epoch": epoch, "avg_loss": loss_meter.avg})

@torch.no_grad()
def test(model, testloader, device, loss_function, epoch, generate_imgs, levels, dataset_name):
    global best_loss  # keep track of best loss
    global test_losses
    model.eval()
    loss = 0
    num_samples = 32  # should probably be an argument
    # TODO: add average loss checker here, they use that in code for checkpointing
    loss_meter = AverageMeter('test-avg')
    for x, y in testloader:
        x = x.to(device)
        z, logdet, _, logp = model(preprocess(x))
        loss = loss_function(logp, logdet, x.size())

        # code for rosalinty model
        # log_p_sum, logdet, z_outs = model(x)
        # loss = loss_function(log_p_sum, logdet, x.size())

        loss_meter.update(loss.item())

    if loss_meter.avg < best_loss:
        print(f"New best model found, average loss {loss_meter.avg}")
        checkpoint_state = {
            "model": model.state_dict(),
            "test_loss": loss_meter.avg,
            "epoch": epoch
        }
        os.makedirs("new_checkpoints", exist_ok=True)
        # save the model
        torch.save(checkpoint_state, f"new_checkpoints/best_{dataset_name.lower()}.pth.tar")
        best_loss = loss_meter.avg
    print(f"test epoch complete, result: {loss_meter.avg} bits/dim")
    test_losses.append({"epoch": epoch, "avg_loss": loss_meter.avg})
    x = next(iter(testloader))[0]  # extract first batch of data in order to get channel dimens
    # generate samples after each test (?)
    if(generate_imgs):
        sample_images = generate(model, num_samples, device, shape=x.shape, levels=levels)
        os.makedirs('generated_imgs', exist_ok=True)
        grid = torchvision.utils.make_grid(sample_images, nrow=int(num_samples ** 0.5))
        torchvision.utils.save_image(grid, f"generated_imgs/epoch_{epoch}.png")

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
    x /= 0.6  # attempt to make it brighter
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DD2412 Mini-glow")
    # using CIFAR optimizations as default here
    parser.add_argument('--batch_size', default=64,
                        type=int, help="minibatch size")
    parser.add_argument('--lr', default=1e-4, help="learning rate")
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

    best_loss = float('inf')
    glbl_step = 0
    train_losses = []
    test_losses = []

    #with autograd.detect_anomaly():
    main(parser.parse_args())

    #write statistics to file
    write_arr_to_csv(train_losses, "new_train_losses")
    write_arr_to_csv(test_losses, "new_test_losses")
