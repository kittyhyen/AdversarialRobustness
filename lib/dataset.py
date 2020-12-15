import numpy as np
import os
import random
import torch
import torchvision
from PIL import Image


class MNIST_with_indices(torchvision.datasets.MNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class FashionMNIST_with_indices(torchvision.datasets.FashionMNIST):
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR10_with_indices(torchvision.datasets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class STL10_with_indices(torchvision.datasets.STL10):
    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index



class CIFAR10_C_with_indices(torch.utils.data.Dataset):
    def __init__(self, transform=torchvision.transforms.ToTensor(), root="data/cifar10-c"):
        self.transform = transform
        self.dir_path = root
        self.transform_file_list = os.listdir(self.dir_path)
        self.transform_file_list.remove("labels.npy")

        self.dataset_x = None
        self.dataset_y = np.load(self.dir_path + "/labels.npy")

    def return_transform_file_list(self):
        return self.transform_file_list

    def set_transform_file_as_dataset_x(self, transform_file_name):
        self.dataset_x = np.load(self.dir_path + "/" + transform_file_name)
        return

    def __len__(self):
        return len(self.dataset_y)

    def __getitem__(self, idx):
        x = self.transform(self.dataset_x[idx])
        y = self.dataset_y[idx]
        return x, y, idx


class CIFAR10_with_base(torchvision.datasets.CIFAR10):
    def __init__(self, *args, base_transform=torchvision.transforms.ToTensor(), **kwargs):
        super().__init__(*args, **kwargs)
        self.base_transform = base_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        base_img = self.base_transform(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (base_img, img), target

