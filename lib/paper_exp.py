import math
import torch
import torchvision
import numpy as np
from lib.dataset import CIFAR10_with_base
from lib.vulnerable_util import torch_cat
import torch.nn.functional as F


def return_loader(dataset, train_transform, test_transform, batch_size, test_batch_size,
                  valid_ratio=0.1, random_seed=10, with_base=False):
    # test batch size를 나눈 이유는 test할때는 restart를 넣을거라서 GPU 터지는것을 막기위해 batch size를 줄이기 위함

    # https://hoya012.github.io/blog/DenseNet-Tutorial-2/
    if dataset in ["cifar10"]:
        if with_base:
            _train = CIFAR10_with_base(root='data/cifar10', train=True,
                                       download=True, transform=train_transform)
        else:
            _train = torchvision.datasets.CIFAR10(root='data/cifar10', train=True,
                                                  download=True, transform=train_transform)
        _valid = torchvision.datasets.CIFAR10(root='data/cifar10', train=True,
                                              download=True, transform=test_transform)
        _test = torchvision.datasets.CIFAR10(root='data/cifar10', train=False,
                                             download=True, transform=test_transform)
    elif dataset in ["cifar100"]:
        _train = torchvision.datasets.CIFAR100(root='data/cifar100', train=True,
                                               download=True, transform=train_transform)
        _valid = torchvision.datasets.CIFAR100(root='data/cifar100', train=True,
                                               download=True, transform=test_transform)
        _test = torchvision.datasets.CIFAR100(root='data/cifar100', train=False,
                                              download=True, transform=test_transform)
    elif dataset in ["stl10"]:
        _train = torchvision.datasets.STL10(root='data/stl10', split='train',
                                            download=True, transform=train_transform)
        _valid = torchvision.datasets.STL10(root='data/stl10', split='train',
                                            download=True, transform=test_transform)
        _test = torchvision.datasets.STL10(root='data/stl10', split='test',
                                           download=True, transform=test_transform)
    elif dataset in ["fmnist"]:
        _train = torchvision.datasets.FashionMNIST(root='data/fmnist', train=True,
                                                   download=True, transform=train_transform)
        _valid = torchvision.datasets.FashionMNIST(root='data/fmnist', train=True,
                                                   download=True, transform=test_transform)
        _test = torchvision.datasets.FashionMNIST(root='data/fmnist', train=False,
                                                  download=True, transform=test_transform)
    elif dataset in ["mnist"]:
        _train = torchvision.datasets.MNIST(root='data/mnist', train=True,
                                                   download=True, transform=train_transform)
        _valid = torchvision.datasets.MNIST(root='data/mnist', train=True,
                                                   download=True, transform=test_transform)
        _test = torchvision.datasets.MNIST(root='data/mnist', train=False,
                                                  download=True, transform=test_transform)
    else:
        raise NotImplementedError

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

    return (num_train, train), (num_valid, valid), (num_test, test)


class InfiniteLoader(object):
    def __init__(self, data_loader):
        super().__init__()
        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)

    def get_next_batch(self):
        try:
            contents = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.data_loader)
            contents = next(self.data_iter)

        return contents


class MultiRandomStartPGDEval(object):
    def __init__(self, model, criterion, epsilon=0.3, num_steps=40, step_size=0.01):
        super().__init__()
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def eval(self, x, multiple, y):
        batch_size = y.shape[0]
        y = y.repeat(multiple).detach()

        x = x.repeat(multiple, 1, 1, 1)

        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        final_output = self.model(adv_x)
        final_correct = final_output.max(1)[1].eq(y)
        final_correct_list = torch.split(final_correct, [batch_size for i in range(multiple)], dim=0)

        final_correct = final_correct_list[0]
        for i in range(1, multiple):
            final_correct = final_correct & final_correct_list[i]

        return final_correct


# https://github.com/JHL-HUST/RLFAT/blob/master/cifar_10/nattack.py
class NAttackEval(object):

    box_min = 0.
    box_max = 1.
    box_plus = (box_min + box_max) / 2.
    box_mul = (box_max - box_min) / 2.

    def __init__(self, model, epsilon, n_pop=300, max_iter=200, sigma=0.1, alpha=0.08, check_iter=10):
        super(NAttackEval, self).__init__()
        self.model = model
        self.epsilon = epsilon
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.sigma = sigma
        self.alpha = alpha

        self.check_iter = check_iter

    def _torch_arctanh(self, x, eps=1e-6):
        x *= (1. - eps)
        return (torch.log((1. + x) / (1. - x))) * 0.5

    def _transform(self, x):
        return torch.tanh(x) * self.box_mul + self.box_plus

    def eval(self, x, y):
        with torch.no_grad():
            logits = self.model(x)
        num_classes = logits.shape[1]
        correct = logits.max(1)[1].eq(y)
        final_correct = torch.clone(correct)

        for data_idx in range(x.shape[0]):
            if final_correct[data_idx]:
                input, target = x[data_idx], y[data_idx]
                modify = (torch.randn(1, *x.shape[1:]) * 0.001).to(x.device)  # 어디서 나온 0.001인지 모르겠네

                for iter_idx in range(self.max_iter):
                    n_sample = torch.randn(self.n_pop, *x.shape[1:]).to(x.device)
                    modify_try = modify.repeat(self.n_pop, 1, 1, 1) + self.sigma * n_sample

                    new_img = self._torch_arctanh((input - self.box_plus)/self.box_mul)
                    transformed_new_img = self._transform(new_img)
                    input_img = self._transform(new_img.unsqueeze(0) + modify_try)
                    if iter_idx % self.check_iter == 0:
                        real_input_img = self._transform(new_img + modify)
                        real_dist = real_input_img - transformed_new_img
                        real_clip_dist = torch.clamp(real_dist, -self.epsilon, self.epsilon)
                        real_clip_input = real_clip_dist + transformed_new_img

                        with torch.no_grad():
                            output_real = self.model(real_clip_input)

                        if output_real.max(1)[1].squeeze().item() != target.item() and (torch.abs(real_clip_dist).max() <= self.epsilon):
                            final_correct[data_idx] = False
                            break
                    dist = input_img - transformed_new_img
                    clip_dist = torch.clamp(dist, -self.epsilon, self.epsilon)
                    clip_input = (clip_dist + transformed_new_img).reshape(self.n_pop, *x.shape[1:])

                    target_onehot = torch.zeros((1, num_classes)).to(x.device)
                    target_onehot[0][target.item()] = 1.

                    with torch.no_grad():
                        outputs = self.model(clip_input)
                    target_onehot = target_onehot.repeat(self.n_pop, 1)

                    real = torch.log((target_onehot * outputs).sum(1) + 1e-30)
                    other = torch.log(((1. - target_onehot) * outputs - target_onehot * 10000.).max(1)[0] + 1e-30)

                    loss1 = torch.clamp(real - other, 0., 1000)

                    reward = 0.5 * loss1
                    reward = -reward

                    a = (reward - torch.mean(reward)) / (torch.std(reward) + 1e-7)

                    modify = modify + (self.alpha / (self.n_pop * self.sigma))*\
                             ((torch.matmul(n_sample.reshape(self.n_pop, -1).T, a)).reshape(*x.shape[1:]))

            else:
                continue

        return final_correct


class CWAttackEval(object):

    def __init__(self, model, k=50, epsilon=8. / 255., num_steps=7, step_size=2. / 255.):
        super(CWAttackEval, self).__init__()
        self.model = model
        self.k = k
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def eval(self, x, y):
        with torch.no_grad():
            clean_output = self.model(x.detach())
            clean_pred = clean_output.max(1)[1].detach()

            num_classes = clean_output.shape[1]
            if y is None:
                y = clean_pred

        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        onehot_label = torch.eye(num_classes)[y].to(x.device)
        min_tensor = torch.tensor(-float("Inf")).to(x.device)
        index_tensor = torch.tensor(list(range(x.shape[0]))).to(x.device)

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)

            correct_logit = output[index_tensor, y]
            wrong_logit = torch.max(torch.where(onehot_label == 1., min_tensor, output), dim=1)[0]

            loss = (-F.relu(correct_logit - wrong_logit + self.k)).sum()

            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        with torch.no_grad():
            final_output = self.model(adv_x)

        return final_output.max(1)[1].eq(y)


class FGSMAttackEval(object):

    def __init__(self, model, criterion, epsilon=8. / 255.):
        super(FGSMAttackEval, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def eval(self, x, y=None):
        if y is None:
            with torch.no_grad():
                clean_output = self.model(x.detach())
                clean_pred = clean_output.max(1)[1].detach()
            y = clean_pred

        adv_x = x.clone().detach()
        adv_x.requires_grad = True
        output = self.model(adv_x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        adv_x_grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

        adv_x = torch.clamp(adv_x + self.epsilon * torch.sign(adv_x_grad), 0, 1).detach()

        with torch.no_grad():
            final_output = self.model(adv_x)

        return final_output.max(1)[1].eq(y)


class RFGSMAttackEval(object):

    def __init__(self, model, criterion, epsilon=8. / 255.):
        super(RFGSMAttackEval, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon

    def eval(self, x, y=None):
        if y is None:
            with torch.no_grad():
                clean_output = self.model(x.detach())
                clean_pred = clean_output.max(1)[1].detach()
            y = clean_pred

        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        adv_x.requires_grad = True
        output = self.model(adv_x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        adv_x_grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

        adv_x = torch.clamp(adv_x + self.epsilon * torch.sign(adv_x_grad), 0, 1).detach()

        with torch.no_grad():
            final_output = self.model(adv_x)

        return final_output.max(1)[1].eq(y)


class PGDAttackEval(object):

    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(),
                 epsilon=8. / 255., num_steps=7, step_size=2. / 255.):
        super(PGDAttackEval, self).__init__()
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def eval(self, x, y=None):
        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if y is None:
            y = self.model(x).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        with torch.no_grad():
            final_output = self.model(adv_x)

        return final_output.max(1)[1].eq(y)


class PGDSanityTest(object):
    def __init__(self, source_model, target_model, criterion=torch.nn.CrossEntropyLoss(), epsilon=0.3, num_steps=40, step_size=0.01):
        super().__init__()
        self.source_model = source_model    # 이 모델로 공격을 만들어서
        self.target_model = target_model    # 이 모델으로 성능 측정
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def eval(self, x, y=None):
        total_correct = None

        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if y is None:
            y = self.source_model(x).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output_from_source = self.source_model(adv_x)
            loss = self.criterion(output_from_source, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

            with torch.no_grad():
                output_from_target = self.target_model(adv_x)
                correct = output_from_target.max(1)[1].eq(y).unsqueeze(dim=1)
            total_correct = torch_cat(total_correct, correct, dim=1)

        return total_correct

