import numpy as np
import torch
import torchvision
from PIL import Image
from lib.resnet_madry import PreActResNet, PreActBasicBlock
import torchvision.transforms.functional as F


class PairFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img1, img2):
        if torch.rand(1) < self.p:
            return F.hflip(img1), F.hflip(img2)
        return img1, img2

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class CIFAR10_HF(torchvision.datasets.CIFAR10):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = 32
        self.pair_flip = PairFlip(p=p)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        hf_img, _ = self.pair_flip(img, img)
        hf_img = self.to_tensor(hf_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return hf_img, target


class CIFAR10_RSCHFPair(torchvision.datasets.CIFAR10):
    def __init__(self, *args, p=0.5, pure_transform="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = 32
        self.rsc = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.))
        if pure_transform == "none":
            self.pure_transform = None
        else:
            self.pure_transform = torchvision.transforms.Resize(int(self.img_size*0.75))
        self.pair_flip = PairFlip(p=p)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        rsc_img = self.rsc(img)

        if self.pure_transform is not None:
            img = self.pure_transform(img)

        hf_img, rschf_img = self.pair_flip(img, rsc_img)
        hf_img, rschf_img = self.to_tensor(hf_img), self.to_tensor(rschf_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (hf_img, rschf_img), target


class CIFAR10_TwoRSCHFPair(torchvision.datasets.CIFAR10):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = 32
        self.rsc = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.))
        self.pair_flip = PairFlip(p=p)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        rsc_img1 = self.rsc(img)
        rsc_img2 = self.rsc(img)

        rschf_img1, rschf_img2 = self.pair_flip(rsc_img1, rsc_img2)
        rschf_img1, rschf_img2 = self.to_tensor(rschf_img1), self.to_tensor(rschf_img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (rschf_img1, rschf_img2), target


class CIFAR100_RSCHFPair(torchvision.datasets.CIFAR100):
    def __init__(self, *args, p=0.5, pure_transform="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = 32
        self.rsc = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.))
        self.pair_flip = PairFlip(p=p)
        if pure_transform == "none":
            self.pure_transform = None
        else:
            self.pure_transform = torchvision.transforms.Resize(int(self.img_size*0.75))
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        rsc_img = self.rsc(img)

        if self.pure_transform is not None:
            img = self.pure_transform(img)

        hf_img, rschf_img = self.pair_flip(img, rsc_img)
        hf_img, rschf_img = self.to_tensor(hf_img), self.to_tensor(rschf_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (hf_img, rschf_img), target


class STL10_RSCHFPair(torchvision.datasets.STL10):
    def __init__(self, *args, p=0.5, pure_transform="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = 96
        self.rsc = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.))
        if pure_transform == "none":
            self.pure_transform = None
        else:
            self.pure_transform = torchvision.transforms.Resize(int(self.img_size*0.75))
        self.pair_flip = PairFlip(p=p)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        rsc_img = self.rsc(img)

        if self.pure_transform is not None:
            img = self.pure_transform(img)

        hf_img, rschf_img = self.pair_flip(img, rsc_img)
        hf_img, rschf_img = self.to_tensor(hf_img), self.to_tensor(rschf_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (hf_img, rschf_img), target


class STL10_HF(torchvision.datasets.STL10):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = 96
        self.pair_flip = PairFlip(p=p)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        hf_img, _ = self.pair_flip(img, img)
        hf_img = self.to_tensor(hf_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return hf_img, target


class ImageFolder_RSCHFPair(torchvision.datasets.ImageFolder):
    def __init__(self, *args, img_size=64, p=0.5, pure_transform="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.rsc = torchvision.transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.))
        if pure_transform == "none":
            self.pure_transform = None
        else:
            self.pure_transform = torchvision.transforms.Resize(int(img_size*0.75))
        self.pair_flip = PairFlip(p=p)
        self.to_tensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        img = sample
        rsc_img = self.rsc(img)

        if self.pure_transform is not None:
            img = self.pure_transform(img)

        hf_img, rschf_img = self.pair_flip(img, rsc_img)
        hf_img, rschf_img = self.to_tensor(hf_img), self.to_tensor(rschf_img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (hf_img, rschf_img), target


def return_loader(dataset, batch_size, test_batch_size, valid_ratio=0.1, random_seed=10,
                  transform="rschf", pure_transform="none"):
    # test batch size를 나눈 이유는 test할때는 restart를 넣을거라서 GPU 터지는것을 막기위해 batch size를 줄이기 위함

    # https://hoya012.github.io/blog/DenseNet-Tutorial-2/
    if dataset in ["cifar10"]:
        if transform == "rschf":
            _train = CIFAR10_RSCHFPair(root='data/cifar10', train=True,
                                       p=0.5, download=True, pure_transform=pure_transform)
        elif transform == "two_rschf":
            _train = CIFAR10_TwoRSCHFPair(root='data/cifar10', train=True,
                                          p=0.5, download=True)
        elif transform == "hf":
            _train = CIFAR10_HF(root='data/cifar10', train=True,
                                p=0.5, download=True)
        else:
            raise NotImplementedError
        _valid = torchvision.datasets.CIFAR10(root='data/cifar10', train=True,
                                              download=True, transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.CIFAR10(root='data/cifar10', train=False,
                                             download=True, transform=torchvision.transforms.ToTensor())
    elif dataset in ["cifar100"]:
        if transform == "rschf":
            _train = CIFAR100_RSCHFPair(root='data/cifar100', train=True,
                                        p=0.5, download=True, pure_transform=pure_transform)
        else:
            raise NotImplementedError
        _valid = torchvision.datasets.CIFAR100(root='data/cifar100', train=True,
                                               download=True, transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.CIFAR100(root='data/cifar100', train=False,
                                             download=True, transform=torchvision.transforms.ToTensor())
    elif dataset in ["stl10"]:
        if transform == "rschf":
            _train = STL10_RSCHFPair(root='data/stl10', split='train',
                                     download=True, p=0.5, pure_transform=pure_transform)
        elif transform == "hf":
            _train = STL10_HF(root='data/stl10', split='train',
                              download=True, p=0.5)
        else:
            raise NotImplementedError
        _valid = torchvision.datasets.STL10(root='data/stl10', split='test',    # test data가 train보다 많아서 씀
                                            download=True, transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.STL10(root='data/stl10', split='test',
                                           download=True, transform=torchvision.transforms.ToTensor())
    elif dataset in ["tiny"]:
        if transform == "rschf":
            _train = ImageFolder_RSCHFPair(root='data/tiny/train', img_size=64, p=0.5, pure_transform=pure_transform)
            #_train = torchvision.datasets.ImageFolder(root='data/tiny/train', transform=torchvision.transforms.ToTensor())
        else:
            raise NotImplementedError
        _valid = torchvision.datasets.ImageFolder(root='data/tiny/train', transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.ImageFolder(root='data/tiny/val', transform=torchvision.transforms.ToTensor())
    else:
        raise NotImplementedError

    if dataset not in ["stl10"]:
        if valid_ratio <= 0.:
            num_train = len(_train)
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
            num_valid = len(_test)
            valid = torch.utils.data.DataLoader(_test, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        else:
            num_train = len(_train)
            indices = list(range(num_train))
            split = int(np.floor(valid_ratio * num_train))

            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            num_train = num_train - split
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                num_workers=2, sampler=train_sampler)

            num_valid = split
            valid = torch.utils.data.DataLoader(_valid, batch_size=batch_size,
                                                num_workers=2, sampler=valid_sampler)

        num_test = len(_test)
        test = torch.utils.data.DataLoader(_test, batch_size=test_batch_size,
                                           shuffle=False, num_workers=2)
    else:
        if valid_ratio <= 0.:
            num_train = len(_train)
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
            num_valid = len(_test)
            valid = torch.utils.data.DataLoader(_test, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

            num_test = len(_test)
            test = torch.utils.data.DataLoader(_test, batch_size=test_batch_size,
                                               shuffle=False, num_workers=2)
        else:
            num_train = len(_train)
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

            num_test = len(_test)
            indices = list(range(num_test))
            split = int(np.floor(valid_ratio * num_test))

            np.random.seed(random_seed)
            np.random.shuffle(indices)

            test_idx, valid_idx = indices[split:], indices[:split]
            test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            num_test = num_test - split
            test = torch.utils.data.DataLoader(_test, batch_size=test_batch_size,
                                               num_workers=2, sampler=test_sampler)

            num_valid = split
            valid = torch.utils.data.DataLoader(_valid, batch_size=batch_size,
                                                num_workers=2, sampler=valid_sampler)

    return (num_train, train), (num_valid, valid), (num_test, test)


def return_loader_adt(dataset, batch_size, test_batch_size, train_transform, valid_ratio=0.1, random_seed=10):
    # transformation이 주어지는 거
    # test batch size를 나눈 이유는 test할때는 restart를 넣을거라서 GPU 터지는것을 막기위해 batch size를 줄이기 위함

    # https://hoya012.github.io/blog/DenseNet-Tutorial-2/
    if dataset in ["cifar10"]:
        _train = torchvision.datasets.CIFAR10(root='data/cifar10', train=True,
                                              download=True, transform=train_transform)
        _valid = torchvision.datasets.CIFAR10(root='data/cifar10', train=True,
                                              download=True, transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.CIFAR10(root='data/cifar10', train=False,
                                             download=True, transform=torchvision.transforms.ToTensor())
    elif dataset in ["cifar100"]:
        _train = torchvision.datasets.CIFAR100(root='data/cifar100', train=True,
                                               download=True, transform=train_transform)
        _valid = torchvision.datasets.CIFAR100(root='data/cifar100', train=True,
                                               download=True, transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.CIFAR100(root='data/cifar100', train=False,
                                              download=True, transform=torchvision.transforms.ToTensor())
    elif dataset in ["stl10"]:
        _train = torchvision.datasets.STL10(root='data/stl10', split='train',
                                            download=True, transform=train_transform)
        _valid = torchvision.datasets.STL10(root='data/stl10', split='test',    # test data가 train보다 많아서 씀
                                            download=True, transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.STL10(root='data/stl10', split='test',
                                           download=True, transform=torchvision.transforms.ToTensor())
    elif dataset in ["tiny"]:
        _train = torchvision.datasets.ImageFolder(root='data/tiny/train', transform=train_transform)
        _valid = torchvision.datasets.ImageFolder(root='data/tiny/train', transform=torchvision.transforms.ToTensor())
        _test = torchvision.datasets.ImageFolder(root='data/tiny/val', transform=torchvision.transforms.ToTensor())
    else:
        raise NotImplementedError

    if dataset not in ["stl10"]:
        if valid_ratio <= 0.:
            num_train = len(_train)
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
            num_valid = len(_test)
            valid = torch.utils.data.DataLoader(_test, batch_size=batch_size,
                                                shuffle=False, num_workers=2)
        else:
            num_train = len(_train)
            indices = list(range(num_train))
            split = int(np.floor(valid_ratio * num_train))

            np.random.seed(random_seed)
            np.random.shuffle(indices)

            train_idx, valid_idx = indices[split:], indices[:split]
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            num_train = num_train - split
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                num_workers=2, sampler=train_sampler)

            num_valid = split
            valid = torch.utils.data.DataLoader(_valid, batch_size=batch_size,
                                                num_workers=2, sampler=valid_sampler)

        num_test = len(_test)
        test = torch.utils.data.DataLoader(_test, batch_size=test_batch_size,
                                           shuffle=False, num_workers=2)
    else:
        if valid_ratio <= 0.:
            num_train = len(_train)
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)
            num_valid = len(_test)
            valid = torch.utils.data.DataLoader(_test, batch_size=batch_size,
                                                shuffle=False, num_workers=2)

            num_test = len(_test)
            test = torch.utils.data.DataLoader(_test, batch_size=test_batch_size,
                                               shuffle=False, num_workers=2)
        else:
            num_train = len(_train)
            train = torch.utils.data.DataLoader(_train, batch_size=batch_size,
                                                shuffle=True, num_workers=2)

            num_test = len(_test)
            indices = list(range(num_test))
            split = int(np.floor(valid_ratio * num_test))

            np.random.seed(random_seed)
            np.random.shuffle(indices)

            test_idx, valid_idx = indices[split:], indices[:split]
            test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)
            valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

            num_test = num_test - split
            test = torch.utils.data.DataLoader(_test, batch_size=test_batch_size,
                                               num_workers=2, sampler=test_sampler)

            num_valid = split
            valid = torch.utils.data.DataLoader(_valid, batch_size=batch_size,
                                                num_workers=2, sampler=valid_sampler)

    return (num_train, train), (num_valid, valid), (num_test, test)


def wide_pre_act_resnet32(width=10, **kwargs):
    layers = [16, 16, 32, 64]
    layers = (np.array(layers) * np.array([1, width, width, width])).tolist()
    return PreActResNetWithFC(PreActBasicBlock, depth=32, layers=layers, **kwargs)