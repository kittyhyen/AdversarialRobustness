# explanation_util에서 성능나왔던 normalized 안한 cam을 pairing 시키는 거 빼냄
# 단순화

import torch
import numpy as np
from lib.resnet_madry import PreActResNet, PreActBasicBlock


class CAMPreActResNet(PreActResNet):
    def __init__(self, *args, **kwargs):
        super(CAMPreActResNet, self).__init__(*args, **kwargs)

    def forward_img_to_conv(self, x):
        if self.input_standardize:
            x = self._input_standardize(x)

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_bn(x)
        x = self.relu(x)

        return x

    def forward_conv_to_logit(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_cam_all(self, x):
        conv_x = self.forward_img_to_conv(x)
        return self.forward_conv_to_cam_all(conv_x)

    def _total_cam_min_max_normalize(self, total_cam, eps=1e-8):
        min_per_cam = torch.unsqueeze(torch.min(torch.reshape(total_cam, (total_cam.shape[0], total_cam.shape[1], -1)),
                                                dim=2, keepdim=True)[0], dim=3).detach()
        max_per_cam = torch.unsqueeze(torch.max(torch.reshape(total_cam, (total_cam.shape[0], total_cam.shape[1], -1)),
                                                dim=2, keepdim=True)[0], dim=3).detach()

        normalized_cam = (total_cam - min_per_cam) / (max_per_cam - min_per_cam + eps)

        return normalized_cam

    def forward_conv_to_cam_all(self, conv_x, normalize_type=None):
        expanded_conv_x = torch.unsqueeze(conv_x, dim=1).repeat(1, self.num_classes, 1, 1, 1)

        # expanded_w : (batch, class, channel, 1, 1)
        w = self.fc.weight
        expanded_w = torch.reshape(w, (1, w.shape[0], w.shape[1], 1, 1)).repeat(conv_x.shape[0], 1, 1, 1, 1)

        total_cam = torch.sum(torch.mul(expanded_conv_x, expanded_w), dim=2)

        if normalize_type is None:
            return total_cam
        elif normalize_type == "min_max":
            return self._total_cam_min_max_normalize(total_cam)
        else:
            raise NotImplementedError


class AttentionPreActResNet(PreActResNet):
    def __init__(self, *args, **kwargs):
        super(AttentionPreActResNet, self).__init__(*args, **kwargs)

    def forward_img_to_conv(self, x):
        if self.input_standardize:
            x = self._input_standardize(x)

        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.final_bn(x)
        x = self.relu(x)

        return x

    def forward_conv_to_logit(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward_attention(self, x):
        conv_x = self.forward_img_to_conv(x)
        return self.forward_conv_to_attention(conv_x)

    def _attention_min_max_normalize(self, attention, eps=1e-8):
        min_per_attention = torch.unsqueeze(torch.min(torch.reshape(attention, (attention.shape[0], -1)),
                                                      dim=1, keepdim=True)[0], dim=2).detach()
        max_per_attention = torch.unsqueeze(torch.max(torch.reshape(attention, (attention.shape[0], -1)),
                                                      dim=1, keepdim=True)[0], dim=2).detach()

        normalized_attention = (attention - min_per_attention) / (max_per_attention - min_per_attention + eps)

        return normalized_attention

    def forward_conv_to_attention(self, conv_x, normalize_type=None):
        # (B, C, H, W)
        attention = torch.sum(torch.abs(conv_x), dim=1)

        if normalize_type is None:
            return attention
        elif normalize_type == "min_max":
            return self._attention_min_max_normalize(attention)
        else:
            raise NotImplementedError


def wide_cam_pre_act_resnet32(width=10, **kwargs):
    layers = [16, 16, 32, 64]
    layers = (np.array(layers) * np.array([1, width, width, width])).tolist()
    return CAMPreActResNet(PreActBasicBlock, depth=32, layers=layers, **kwargs)


def wide_attention_pre_act_resnet32(width=10, **kwargs):
    layers = [16, 16, 32, 64]
    layers = (np.array(layers) * np.array([1, width, width, width])).tolist()
    return AttentionPreActResNet(PreActBasicBlock, depth=32, layers=layers, **kwargs)

