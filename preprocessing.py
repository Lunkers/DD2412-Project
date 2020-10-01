import torchvision
import torch
import os
import future
import numpy

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
        self.path2ImageNetZip = os.getcwd() + "/ImageNet_zip"
        self.path2ImageNetRoot = "/opt/anaconda3/envs/ADL/lib/python3.7/site-packages/torchvision/datasets"

    def load_data(self):

        if not os.path.isdir(self.path2data):
            os.mkdir(self.path2data)

        if not os.path.isfile(self.path2data + "/cifar-10-python.tar.gz"):
            print("Preparing CIFAR10\n")
            cifar10 = torchvision.datasets.CIFAR10(
                root=self.path2data, download=True)
            data_loader_cifar10 = torch.utils.data.DataLoader(
                cifar10, batch_size=4, shuffle=True)
        else:
            print("CIFAR10 done.\n")

        if not os.path.isfile(self.path2data + "/ILSVRC2012_img_val.tar"):
            print("Preparing ImageNet\n")
            ImageNet = torchvision.datasets.ImageNet(
                root=self.path2data)
            data_loader_ImageNet = torch.utils.data.DataLoader(
                ImageNet, batch_size=4, shuffle=True)
        else:
            print("ImageNet done.")

        return cifar10, data_loader_cifar10, ImageNet, data_loader_ImageNet


if __name__ == "__main__":
    DL = Dataloader()
    cifar10, data_loader_cifar10, ImageNet, data_loader_ImageNet = DL.load_data()
