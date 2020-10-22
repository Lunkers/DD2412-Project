import torchvision
import torch
import os
import future
import numpy as np
import matplotlib.pyplot as plt

transform_cifar = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(
    degrees=0, translate=(0.1, 0.1)), torchvision.transforms.ToTensor()])
transform_mnist = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(
    degrees=0, translate=(0.1, 0.1)), torchvision.transforms.ToTensor()])

class Dataloader():
    def __init__(self):
        self.path2root = os.getcwd()
        self.path2data = os.getcwd() + "/data"

    def load_data(self, batch_size_cifar=4, batch_size_mnist=4):

        if not os.path.isdir(self.path2data):
            os.mkdir(self.path2data)

        traincifar10 = torchvision.datasets.CIFAR10(
            root=self.path2data, train=True, download=True, transform=transform_cifar)
        trainloader_cifar10 = torch.utils.data.DataLoader(
            traincifar10, batch_size=batch_size_cifar, shuffle=True)

        testcifar10 = torchvision.datasets.CIFAR10(
            root=self.path2data, train=False, download=True, transform=transform_cifar)
        testloader_cifar10 = torch.utils.data.DataLoader(
            testcifar10, batch_size=batch_size_cifar, shuffle=False)

        trainmnist = torchvision.datasets.MNIST(
            self.path2data, train=True, download=True, transform=transform_mnist)
        trainloader_mnist = torch.utils.data.DataLoader(
            trainmnist, batch_size=batch_size_mnist, shuffle=True)

        testmnist = torchvision.datasets.MNIST(
            root=self.path2data, train=False, download=True, transform=transform_mnist)
        testloader_mnist = torch.utils.data.DataLoader(
            testmnist, batch_size=batch_size_mnist, shuffle=False)

        return trainloader_cifar10, testloader_cifar10, trainloader_mnist, testloader_mnist

    def load_subset_of_cifar(self, batch_size_cifar=4, subset_train_size=1000, subset_test_size=100):
        if not os.path.isdir(self.path2data):
            os.mkdir(self.path2data)

        traincifar10 = torchvision.datasets.CIFAR10(
            root=self.path2data, train=True, download=True, transform=transform_cifar)
        testcifar10 = torchvision.datasets.CIFAR10(
            root=self.path2data, train=False, download=True, transform=transform_cifar)

        traincifar10 = torch.utils.data.Subset(traincifar10, range(subset_train_size))
        testcifar10 = torch.utils.data.Subset(testcifar10, range(subset_test_size))

        trainloader_cifar10 = torch.utils.data.DataLoader(
            traincifar10, batch_size=batch_size_cifar, shuffle=True)
        testloader_cifar10 = torch.utils.data.DataLoader(
            testcifar10, batch_size=batch_size_cifar, shuffle=False)

        return trainloader_cifar10, testloader_cifar10


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    DL = Dataloader()
    train_cifar, test_cifar, train_mnist, test_mnist = DL.load_data()

    # dataiter = iter(train_cifar)
    # images, labels = dataiter.next()
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # dataiter = iter(train_mnist)
    # images, labels = dataiter.next()
    # plt.imshow(images[0].reshape(28,28), cmap="gray")
    # plt.show()
    # print("Number of batches with batch size 4 is: %s" % (len(dataiter)))
