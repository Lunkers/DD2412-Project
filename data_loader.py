import shutil
import os
import numpy as np
import torchvision
import PIL
import torch


class DataLoader():

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

    def Load_ImageNet(self):
        self.extract_data()
        transform_ = torchvision.transforms.Compose([torchvision.transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1)), torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0, 0, 0), (1, 1, 1))])

        os.chdir(self.path2imageNet_train)
        transformed_images_train = []
        for fileName in self.filenames_train:
            im = PIL.Image.open(fileName)
            transformed_images_train.append(transform_(im))

        trainloader_ImageNet = torch.utils.data.DataLoader(
            transformed_images_train, batch_size=4)

        os.chdir(self.path2imageNet_val)
        transformed_images_val = []
        for fileName in self.filenames_val:
            im = PIL.Image.open(fileName)
            transformed_images_val.append(transform_(im))

        testloader_ImageNet = torch.utils.data.DataLoader(
            transformed_images_val, batch_size=4)

        return trainloader_ImageNet, testloader_ImageNet


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    DL = DataLoader()
    trainloader_ImageNet, testloader_ImageNet = DL.Load_ImageNet()

    dataiter = iter(testloader_ImageNet)
    images = dataiter.next()
    imshow(torchvision.utils.make_grid(images))
