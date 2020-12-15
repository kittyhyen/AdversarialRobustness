import torch
import torch.nn.functional as F

def pairwise_inner(inputs1, inputs2, detach=True):
    # (N, D) -> N: batch, D: dimension
    if detach:
        inputs1, inputs2 = inputs1.detach(), inputs2.detach()

    inner = torch.matmul(inputs1, inputs2.T)

    return inner


def pairwise_cosine_similarity(inputs1, inputs2, eps=1e-8, detach=True):
    # (N, D) -> N: batch, D: dimension
    if detach:
        inputs1, inputs2 = inputs1.detach(), inputs2.detach()

    inner = torch.matmul(inputs1, inputs2.T)

    norm_inputs1 = torch.norm(inputs1, p=2, dim=1)
    norm_inputs2 = torch.norm(inputs2, p=2, dim=1)

    tile_norm1 = torch.unsqueeze(norm_inputs1, 1).repeat(1, inputs2.shape[0])
    tile_norm2 = torch.unsqueeze(norm_inputs2, 0).repeat(inputs1.shape[0], 1)

    denom = torch.mul(tile_norm1, tile_norm2)

    return inner / (denom + eps)


def pairwise_angular_distance(inputs1, inputs2, eps=1e-8, detach=True):
    return 1. - torch.abs(pairwise_cosine_similarity(inputs1, inputs2, eps=eps, detach=detach))


def batch_hard_angular_negative_indices(anchor_features, sample_features, anchor_labels, sample_labels, eps=1e-8):
    # 가장 가까운 negative
    # angular distance: 0 ~ 1
    # 구현상은 abs(cos)가 가장 큰거 고르기로 구현
    # abs(cos): 0 ~ 1
    pair_dist = torch.abs(pairwise_cosine_similarity(anchor_features, sample_features, eps=1e-8))

    ################## label mask ##################
    anchor_labels = torch.unsqueeze(anchor_labels, 1)
    sample_labels = torch.unsqueeze(sample_labels, 0)
    label_mask = (anchor_labels != sample_labels).float()
    ################################################

    masked_pair_dist = torch.mul(pair_dist, label_mask)

    return torch.max(masked_pair_dist, dim=1)[1]


def batch_hard_angular_positive_negative_indices(adv_features, features, adv_labels, labels, eps=1e-8):
    # 가장 먼 positive와 가장 가까운 negative
    # angular distance: 0 ~ 1
    # 0을 곱해야하기 때문에
    # pos는 1-abs(cos)이 가장 큰 것을 고르고
    # neg는 abs(cos)이 가장 큰 것을 고름
    abs_cos_pair_dist = torch.abs(pairwise_cosine_similarity(adv_features, features, eps=1e-8))

    ################## label mask ##################
    adv_labels = torch.unsqueeze(adv_labels, 1)
    labels = torch.unsqueeze(labels, 0)

    positive_label_mask = (adv_labels == labels).float()
    negative_label_mask = (adv_labels != labels).float()
    ################################################

    positive_masked_pair_dist = torch.mul(1. - abs_cos_pair_dist, positive_label_mask)
    negative_masked_pair_dist = torch.mul(abs_cos_pair_dist, negative_label_mask)

    return torch.max(positive_masked_pair_dist, dim=1)[1], torch.max(negative_masked_pair_dist, dim=1)[1]


class AngularDistance(torch.nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(AngularDistance, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return 1. - torch.abs(torch.nn.functional.cosine_similarity(x1, x2, self.dim, self.eps))


class AngularTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.03, eps=1e-8, reduction='batch_mean'):
        super(AngularTripletLoss, self).__init__()
        self.dist_criterion = AngularDistance(dim=1, eps=eps)
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):

        a_p_dist = self.dist_criterion(anchor, positive)
        a_n_dist = self.dist_criterion(anchor, negative)

        if self.reduction == 'batch_mean':
            return torch.mean(torch.max(a_p_dist - a_n_dist + self.margin, torch.zeros(anchor.shape[0]).float().to(anchor.device)))
        else:
            raise NotImplementedError


class AngularInterLoss(torch.nn.Module):
    def __init__(self, margin=0.03, eps=1e-8, reduction='batch_mean'):
        super(AngularInterLoss, self).__init__()
        self.dist_criterion = AngularDistance(dim=1, eps=eps)
        self.margin = margin
        self.reduction = reduction

    def forward(self, features):
        center = torch.mean(features, dim=0, keepdim=True)
        total_dist = self.dist_criterion(center, features)

        if self.reduction == 'batch_mean':
            return torch.mean(F.relu(self.margin - total_dist))
        else:
            raise NotImplementedError


class CoralLoss(torch.nn.Module):
    def __init__(self):
        super(CoralLoss, self).__init__()

    def _return_cov(self, x):
        # x's shape: (data, feature)
        num_data = x.shape[0]
        u = torch.unsqueeze(torch.mean(x, dim=0), 0).repeat(num_data, 1)
        x_m_u = x - u
        return torch.matmul(x_m_u.T, x_m_u) / (num_data-1)

    def forward(self, inputs1, inputs2):
        cov1 = self._return_cov(inputs1)
        cov2 = self._return_cov(inputs2)

        return torch.mean(torch.abs(cov1 - cov2))


class MMDLoss(torch.nn.Module):
    def __init__(self):
        super(MMDLoss, self).__init__()

    def forward(self, inputs1, inputs2):
        # inputs's shape: (data, feature)
        mean_inputs1 = torch.mean(inputs1, dim=0)
        mean_inputs2 = torch.mean(inputs2, dim=0)

        return torch.mean(torch.abs(mean_inputs1 - mean_inputs2))


class MarginLoss(torch.nn.Module):
    def __init__(self, c_lr=0.1, num_classes=10, num_features=10):
        super(MarginLoss, self).__init__()
        self.soft_plus = torch.nn.Softplus()
        self.c_lr = c_lr
        self.num_classes = num_classes
        self.num_features = num_features
        self.center_features = torch.zeros(num_classes, num_features)

    def _center_update(self, features, labels):
        for class_i in range(self.num_classes):
            class_indices = torch.where(labels == class_i)[0]
            num_class_data = class_indices.shape[0]
            if num_class_data != 0:
                c_m_f = torch.unsqueeze(self.center_features[class_i], 0).repeat(num_class_data, 1) - torch.index_select(features, 0, class_indices)
                diff = torch.sum(c_m_f, dim=0) / (1 + num_class_data)

                self.center_features[class_i] -= self.c_lr * diff

    def forward(self, features1, features2, labels1, labels2):
        # inputs's shape: (data, feature)
        # softplus라서 자기 클래스의 dist가 0이나오더라도 grad가 0이 아니기때문에 masking 필요함
        self.center_features = self.center_features.to(features1.device)

        total_features = torch.cat((features1, features2), dim=0)
        total_labels = torch.cat((labels1, labels2), dim=0)
        expand_total_features = torch.unsqueeze(total_features, 1).repeat(1, self.num_classes, 1)

        centers_per_data = torch.index_select(self.center_features, 0, total_labels)
        expand_centers_per_data = torch.unsqueeze(centers_per_data, 1).repeat(1, self.num_classes, 1)

        all_classes_centers_per_data = torch.unsqueeze(self.center_features, 0).repeat(total_features.shape[0], 1, 1)

        anchor_m_pos = expand_total_features - expand_centers_per_data
        anchor_m_all = expand_total_features - all_classes_centers_per_data

        loss_matrix = self.soft_plus(torch.sum(torch.abs(anchor_m_pos), dim=2) - torch.sum(torch.abs(anchor_m_all), dim=2))

        #################### mask ####################
        total_one_hot_labels = torch.eye(self.num_classes)[total_labels]
        total_labels_mask = (total_one_hot_labels == 0).float().to(features1.device)
        ##############################################

        final_loss_matrix = total_labels_mask * loss_matrix

        self._center_update(total_features.detach(), total_labels.detach())

        return torch.mean(torch.sum(final_loss_matrix, 1)) / (self.num_features - 1)


class CosineDistanceLoss(torch.nn.Module):
    def __init__(self, eps=1e-8):
        super(CosineDistanceLoss, self).__init__()
        self.eps = eps

    def forward(self, x):
        # x: (num, dim)

        pair_dist = torch.abs(pairwise_cosine_similarity(x, x, eps=self.eps, detach=False))
        mask = (torch.eye(x.shape[0]) == 0).float().to(x.device)
        masked_pair_dist = torch.mul(mask, pair_dist)

        return torch.mean(masked_pair_dist)


class SupervisedContrastiveLoss(torch.nn.Module):
    # https://github.com/HobbitLong/SupContrast
    def __init__(self, temp=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temp = temp

    def forward(self, inputs, labels):
        # inputs: (num, dim)
        # labels: (num)
        labels = torch.reshape(labels, (-1, 1))

        pair_inner = torch.div(pairwise_inner(inputs, inputs, detach=False), self.temp)
        exp_logits = torch.exp(pair_inner - torch.max(pair_inner, dim=1, keepdim=True)[0].detach())

        other_mask = (torch.eye(inputs.shape[0]) == 0).float().to(inputs.device)
        positive_mask = torch.eq(labels, labels.T).float().detach()

        sum_exp_logits = torch.sum(exp_logits * other_mask, dim=1, keepdim=True)
        softmax_logits = torch.log(exp_logits) - torch.log(sum_exp_logits)
        masked_softmax_logits = other_mask * positive_mask * softmax_logits

        return -torch.sum(torch.mean(masked_softmax_logits, dim=1))


class EuclideanTripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2, reduction='batch_mean'):
        super(EuclideanTripletLoss, self).__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, anchor, positive, negative):

        a_p_dist = torch.sum((anchor - positive).pow(2), dim=1)
        a_n_dist = torch.sum((anchor - negative).pow(2), dim=1)

        if self.reduction == 'batch_mean':
            return torch.mean(F.relu(a_p_dist - a_n_dist + self.margin))
        else:
            raise NotImplementedError