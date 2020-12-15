import torch
import torch.nn.functional as F
import torchvision
import numpy as np


# https://github.com/facebookresearch/mixup-cifar10/blob/master/train.py

class MixUpGenerator(object):
    def __init__(self, beta=1, prob=0., scale=(0., 1.)):
        super().__init__()
        self.beta = beta
        self.prob = prob
        self.scale = scale

    def generate(self, inputs, labels):
        p = np.random.uniform(0., 1.)
        if p < self.prob:
            mixup_inputs = inputs.clone()
            mixup_labels = labels.clone()

            shuffle_indices = torch.randperm(inputs.shape[0])

            mixup_inputs = mixup_inputs[shuffle_indices]
            mixup_labels = mixup_labels[shuffle_indices]

            lam = np.random.beta(self.beta, self.beta)
            lam = self.scale[0] + (self.scale[1] - self.scale[0]) * lam

            mixup_inputs = mixup_inputs * (1. - lam) + inputs * lam

            return mixup_inputs, (1. - lam, mixup_labels), (lam, labels)
        else:
            return inputs, (1., labels), (0., labels)


class MixUpGenerator_v1(object):
    # 만드는 것만 관심있는 generator
    # batch 내의 data들이 섞이는 정도를 다 따로
    def __init__(self, beta=1, prob=0., scale=(0., 1.)):
        super().__init__()
        self.beta = beta
        self.prob = prob
        self.scale = scale

    def generate(self, inputs):
        p = np.random.uniform(0., 1.)
        if p < self.prob:
            mixup_inputs = inputs.clone()

            shuffle_indices = torch.randperm(inputs.shape[0])
            mixup_inputs = mixup_inputs[shuffle_indices]

            lam = np.random.beta(self.beta, self.beta, size=inputs.shape[0])
            lam = self.scale[0] + (self.scale[1] - self.scale[0]) * lam
            lam = torch.tensor(lam).reshape(-1, 1, 1, 1).float().to(inputs.device)

            mixup_inputs = mixup_inputs * (1. - lam) + inputs * lam

            return mixup_inputs
        else:
            return inputs

