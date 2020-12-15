import torch
import torch.nn.functional as F


def return_classification_loss(criterion, outputs, labels, label_sm):
    if label_sm is None:
        return criterion(outputs, labels)
    else:
        assert label_sm < 0.1

        exp_labels = torch.reshape(labels, (-1, 1))
        one_hot_labels = torch.zeros_like(outputs).float() + label_sm
        one_hot_labels = one_hot_labels.scatter(1, exp_labels, 1-9*label_sm)

        kl_criterion = torch.nn.KLDivLoss(reduction='batchmean')

        return kl_criterion(torch.log_softmax(outputs, dim=1), one_hot_labels)


class SmoothingLoss(torch.nn.Module):
    def __init__(self, p_for_true=0.9, num_classes=10):
        super(SmoothingLoss, self).__init__()

        assert 1/num_classes < p_for_true <= 1
        assert num_classes > 1

        self.num_classes = num_classes
        self.p_for_true = p_for_true
        self.p_for_false = (1. - p_for_true)/(num_classes - 1)

    def forward(self, logits, labels):
        if self.p_for_true == 1:
            return F.cross_entropy(logits, labels)
        else:
            exp_labels = torch.reshape(labels, (-1, 1))
            one_hot_labels = torch.zeros_like(logits).float() + self.p_for_false
            one_hot_labels = one_hot_labels.scatter(1, exp_labels, self.p_for_true)

            return F.kl_div(torch.log(one_hot_labels), torch.softmax(logits, dim=1))


