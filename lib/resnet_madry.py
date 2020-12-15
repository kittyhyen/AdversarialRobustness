# resnet_imagenet 및 https://keras.io/examples/cifar10_resnet/ 따름
# 다만 madry의 경우 사이즈 조정할때 padding을 사용했는데 여기선 1*1을 사용함
# 사이즈 맞출때 average를 내는 모델(madry) vs 띄어넘는 모델의 차이가 있긴 함
# madry 모델은 형식은 resnet32

# full pre-activation의 경우 madry가 구현한게 논문이랑 정확한거같다

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    # https://github.com/MadryLab/cifar10_challenge/blob/master/model.py#L124
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, special=False):
        super(PreActBasicBlock, self).__init__()
        self.special = special  # pre act 논문의 appendix에서 언급하는 special attention을 pay attention했다는 부분

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.bn1 = norm_layer(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)

        self.bn2 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        if self.special:
            out = self.bn1(x)
            out = self.relu(out)
            orig_x = out
        else:
            orig_x = x
            out = self.bn1(x)
            out = self.relu(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        identity = self.downsample(orig_x) if self.downsample is not None else x

        out += identity

        return out


class ResNet(nn.Module):

    def __init__(self, block, depth, num_classes=10, layers=[16, 16, 32, 64], zero_init_residual=False,
                 norm_layer=None, input_standardize=False):
        super(ResNet, self).__init__()

        self.num_classes = num_classes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_standardize = input_standardize

        self.inplanes = layers[0]

        self.depth = depth
        assert ((depth-2) % 6) == 0
        num_res_blocks = int((depth-2) / 6)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, layers[1], num_res_blocks)
        self.layer2 = self._make_layer(block, layers[2], num_res_blocks, stride=2)
        self.layer3 = self._make_layer(block, layers[3], num_res_blocks, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layers[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _input_standardize(self, x):
        # https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
        original_shape = x.shape

        x = x.reshape(x.shape[0], -1)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        x_std = torch.max((torch.ones_like(x_std) / math.sqrt(float(x.shape[1]))).to(x.device),
                          x_std)

        x = ((x - x_mean) / x_std).reshape(original_shape)

        return x

    def forward(self, x):
        x = self.forward_feature_extractor(x)
        x = self.forward_classifier(x)

        return x

    def forward_feature_extractor(self, x):
        if self.input_standardize:
            x = self._input_standardize(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_classifier(self, x):
        x = self.fc(x)

        return x


class PreActResNet(nn.Module):
    def __init__(self, block, depth,
                 num_classes=10, layers=(16, 16, 32, 64), zero_init_residual=False,
                 norm_layer=None, input_standardize=False):
        super(PreActResNet, self).__init__()

        self.num_classes = num_classes
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.input_standardize = input_standardize

        self.inplanes = layers[0]

        self.depth = depth
        assert ((depth-2) % 6) == 0
        num_res_blocks = int((depth-2) / 6)

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.layer1 = self._make_layer(block, layers[1], num_res_blocks, special=True)
        self.layer2 = self._make_layer(block, layers[2], num_res_blocks, stride=2, special=False)
        self.layer3 = self._make_layer(block, layers[3], num_res_blocks, stride=2, special=False)
        self.final_bn = norm_layer(layers[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(layers[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, special=False):
        norm_layer = self._norm_layer
        downsample = None

        if self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            norm_layer=norm_layer, special=special))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, special=False))

        return nn.Sequential(*layers)

    def _input_standardize(self, x):
        # https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization
        original_shape = x.shape

        x = x.reshape(x.shape[0], -1)
        x_mean = x.mean(dim=1, keepdim=True)
        x_std = x.std(dim=1, keepdim=True)
        x_std = torch.max((torch.ones_like(x_std) / math.sqrt(float(x.shape[1]))).to(x.device),
                          x_std)

        x = ((x - x_mean) / x_std).reshape(original_shape)

        return x

    def forward(self, x):
        x = self.forward_feature_extractor(x)
        x = self.forward_classifier(x)

        return x

    def forward_feature_extractor(self, x):
        if self.input_standardize:
            x = self._input_standardize(x)

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_classifier(self, x):
        x = self.fc(x)

        return x


def resnet20(**kwargs):
    return ResNet(BasicBlock, depth=20, layers=[16, 16, 32, 64], **kwargs)


def resnet32(**kwargs):
    return ResNet(BasicBlock, depth=32, layers=[16, 16, 32, 64], **kwargs)


def resnet44(**kwargs):
    return ResNet(BasicBlock, depth=44, layers=[16, 16, 32, 64], **kwargs)


def resnet56(**kwargs):
    return ResNet(BasicBlock, depth=56, layers=[16, 16, 32, 64], **kwargs)


def resnet110(**kwargs):
    return ResNet(BasicBlock, depth=110, layers=[16, 16, 32, 64], **kwargs)


def wide_resnet32(width=10, **kwargs):
    layers = [16, 16, 32, 64]
    layers = (np.array(layers) * np.array([1, width, width, width])).tolist()
    return ResNet(BasicBlock, depth=32, layers=layers, **kwargs)


def wide_pre_act_resnet32(width=10, **kwargs):
    layers = [16, 16, 32, 64]
    layers = (np.array(layers) * np.array([1, width, width, width])).tolist()
    return PreActResNet(PreActBasicBlock, depth=32, layers=layers, **kwargs)


def wide_pre_act_resnet38(width=10, **kwargs):
    layers = [16, 16, 32, 64]
    layers = (np.array(layers) * np.array([1, width, width, width])).tolist()
    return PreActResNet(PreActBasicBlock, depth=38, layers=layers, **kwargs)
