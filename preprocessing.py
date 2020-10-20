import torchvision
import torch
import os
import future
import numpy as np
import matplotlib.pyplot as plt
import PIL
import shutil


class Dataloader():
    def __init__(self):
        self.path2root = os.getcwd()
        self.path2data = os.getcwd() + "/data"
        self.path2imageNet_val = self.path2data + "/valid_32x32"
        self.path2imageNet_train = self.path2data + "/train_32x32"

        self.filenames_train = None
        self.filenames_val = None

    def extract_data(self):
        os.chdir(self.path2data)

        if not os.path.isdir(self.path2imageNet_val):
            unpack = [shutil.unpack_archive(filename, self.path2data)
                      for filename in os.listdir() if filename.endswith("tar")]

        self.filenames_train = os.listdir(self.path2imageNet_train)
        self.filenames_val = os.listdir(self.path2imageNet_val)

        return 0

    def load_data(self, batch_size_cifar=4, batch_size_ImageNet=4):
        """
        Loading CIFAR10
        """
        if not os.path.isdir(self.path2data):
            os.mkdir(self.path2data)

        transform_cifar = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))])

        traincifar10 = torchvision.datasets.CIFAR10(
            root=self.path2data, train=True, download=True, transform=transform_cifar)
        trainloader_cifar10 = torch.utils.data.DataLoader(
            traincifar10, batch_size=batch_size_cifar, shuffle=True)

        testcifar10 = torchvision.datasets.CIFAR10(
            root=self.path2data, train=False, download=True, transform=transform_cifar)
        testloader_cifar10 = torch.utils.data.DataLoader(
            testcifar10, batch_size=batch_size_cifar, shuffle=False)

        """
        Loading ImageNet
        """
        self.extract_data()
        transform_ImageNet = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))])

        os.chdir(self.path2imageNet_train)
        transformed_images_train = []
        for fileName in self.filenames_train:
            im = PIL.Image.open(fileName)
            transformed_images_train.append(transform_ImageNet(im))

        trainloader_ImageNet = torch.utils.data.DataLoader(
            transformed_images_train, batch_size=batch_size_ImageNet)

        os.chdir(self.path2imageNet_val)
        transformed_images_val = []
        for fileName in self.filenames_val:
            im = PIL.Image.open(fileName)
            transformed_images_val.append(transform_ImageNet(im))

        testloader_ImageNet = torch.utils.data.DataLoader(
            transformed_images_val, batch_size=batch_size_ImageNet)

        return trainloader_cifar10, testloader_cifar10, trainloader_ImageNet, testloader_ImageNet


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    DL = Dataloader()
    train_cifar, test_cifar, train_ImageNet, test_ImageNet = DL.load_data()

    # dataiter = iter(train_cifar)
    # images, labels = dataiter.next()
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #
    # imshow(torchvision.utils.make_grid(images))
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

    # dataiter = iter(test_ImageNet)
    # images = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))

    # print("Number of batches with batch size 4 is: %s" % (len(dataiter)))
