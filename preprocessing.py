import torchvision
import torch
import os
import future
import numpy
import zipfile


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
            print("CIFAR10 already done.\n")

        if len(os.listdir(self.path2ImageNetZip)) <= 3:
            print("Unzips ImageNet manually...")
            pathNames = ["/Imagenet32_val_npz.zip",
                         "/Imagenet32_train_npz.zip"]
            for pathName in pathNames:
                with zipfile.ZipFile(self.path2ImageNetZip + pathName, 'r') as zip_ref:
                    zip_ref.extractall(self.path2ImageNetZip)
        else:
            print("Files already unzipped.")

        if not os.path.isfile(self.path2data + "/ImageNet.tar.gz"):
            print("Preparing ImageNet\n")
            ImageNet = torchvision.datasets.ImageNet(
                root=self.path2data)
            data_loader_ImageNet = torch.utils.data.DataLoader(
                ImageNet, batch_size=4, shuffle=True)
        else:
            print("ImageNet already done.")

        return cifar10, data_loader_cifar10, ImageNet, data_loader_ImageNet


if __name__ == "__main__":
    DL = Dataloader()
    cifar10, data_loader_cifar10, ImageNet, data_loader_ImageNet = DL.load_data()
