import torchvision
import torch
import os
import future
import numpy as np
import matplotlib.pyplot as plt

"""
Start by downloading these manually:
http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_devkit_t12.tar.gz
http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train.tar
http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_val.tar

"""


class Dataloader():
    def __init__(self):
        self.path2root = os.getcwd()
        self.path2data = os.getcwd() + "/data"

    def load_data(self, batchSizeIMGNET=4, batchSizeCIFAR=4):

        if not os.path.isdir(self.path2data):
            os.mkdir(self.path2data)

        transform = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))])

        if not os.path.isfile(self.path2data + "/cifar-10-python.tar.gz"):
            print("Preparing CIFAR10\n")
            traincifar10 = torchvision.datasets.CIFAR10(
                root=self.path2data, train=True, download=True, transform=transform)
            trainloader_cifar10 = torch.utils.data.DataLoader(
                traincifar10, batch_size=batchSizeCIFAR, shuffle=True)

            testcifar10 = torchvision.datasets.CIFAR10(
                root=self.path2data, train=False, download=True, transform=transform)
            testloader_cifar10 = torch.utils.data.DataLoader(
                testcifar10, batch_size=batchSizeCIFAR, shuffle=False)
        return trainloader_cifar10, testloader_cifar10

        # if not os.path.isfile(self.path2data + "/ILSVRC2012_img_val.tar"):
        #     print("Preparing ImageNet\n")
        #     trainImageNet = torchvision.datasets.ImageNet(
        #         root=self.path2data, train=True, transform=transform)
        #     trainloader_ImageNet = torch.utils.data.DataLoader(
        #         trainImageNet, batch_size=batchSizeIMGNET, shuffle=True)

        #     testImageNet = torchvision.datasets.ImageNet(
        #         root=self.path2data, train=False, transform=transform)
        #     testloader_ImageNet = torch.utils.data.DataLoader(
        #         testImageNet, batch_size=batchSizeIMGNET, shuffle=False)
        # else:
        #     print("ImageNet done.")

        # return trainloader_cifar10, testloader_cifar10, trainloader_ImageNet, testloader_ImageNet


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    DL = Dataloader()
    train, test = DL.load_data()

    dataiter = iter(train)
    images, labels = dataiter.next()
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
