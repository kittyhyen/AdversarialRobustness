# refer to https://github.com/hendrycks/pre-training/blob/master/robustness/adversarial/attacks.py (PGD)

# 지금 이 코드는 input이 (0,1) 범위에 있을 때로 되있는 것들이 대다수라서 noise 크기로 분리해야될거같긴함

# 그리고 noise clipping을 input으로 하고 있기 때문에 grad_sign이 코드에는 들어가있지만 실제로 l2는 안됨

import torch
import torch.nn.functional as F
from lib.vulnerable_util import torch_cat


def normalize_l2(x):
    """
    Expects x.shape == [N, C, H, W]
    """
    norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
    norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    return x / norm


class Attack(object):
    def __init__(self, model):
        super(Attack).__init__()
        self.model = model

    def generate(self, x, y=None):
        raise NotImplementedError()


class RandomSignAttack(Attack):
    def __init__(self, model, epsilon=0.05):
        super(RandomSignAttack, self).__init__(model)
        self.noise = torch.distributions.normal.Normal(loc=0, scale=1)
        self.epsilon = epsilon

    def generate(self, x, y=None):
        batch_noise = self.noise.sample(x.shape).sign().to(x.device)
        adv_x = torch.clamp(x.detach() + self.epsilon * batch_noise, 0, 1)
        return adv_x


class RandomSignInflationAttack(Attack):
    def __init__(self, model, epsilon=0.3, num_quan=30, inflation_rate=40):
        super().__init__(model)
        self.noise = torch.distributions.normal.Normal(loc=0, scale=1)
        self.epsilon = epsilon
        self.num_quan = num_quan
        assert num_quan > 0
        self.quan_size = epsilon / num_quan
        self.inflation_rate = inflation_rate

    def generate(self, x, y=None, same_mask=True):
        sign_mask = self.noise.sample(x.shape).sign().to(x.device)

        total_adv_x = None
        for i in range(self.inflation_rate):
            if not same_mask:
                sign_mask = self.noise.sample(x.shape).sign().to(x.device)
            adv_x = torch.clamp(x.detach() + self.quan_size * torch.randint_like(sign_mask, low=1, high=self.num_quan) * sign_mask, 0, 1).detach()
            if total_adv_x is None:
                total_adv_x = adv_x
            else:
                total_adv_x = torch.cat((total_adv_x, adv_x), dim=0)

        return total_adv_x


class FGSMAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05):
        super(FGSMAttack, self).__init__(model)
        self.criterion = criterion
        self.epsilon = epsilon

    def generate(self, x, y=None):
        adv_x = x.detach()
        adv_x.requires_grad = True
        output = self.model(adv_x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        adv_x_grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

        adv_x = torch.clamp(adv_x + self.epsilon * torch.sign(adv_x_grad), 0, 1).detach()

        return adv_x


class FGSMInflationAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.3, num_quan=30, inflation_rate=40):
        super().__init__(model)
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_quan = num_quan
        assert num_quan > 0
        self.quan_size = epsilon / num_quan
        self.inflation_rate = inflation_rate

    def generate(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

        sign_mask = torch.sign(x_grad.detach())

        total_adv_x = None
        for i in range(self.inflation_rate):
            adv_x = torch.clamp(x.detach() + self.quan_size * torch.randint_like(sign_mask, low=1, high=self.num_quan) * sign_mask, 0, 1).detach()
            if total_adv_x is None:
                total_adv_x = adv_x
            else:
                total_adv_x = torch.cat((total_adv_x, adv_x), dim=0)

        return total_adv_x


class RandomFGSMAttack(Attack):
    def __init__(self, model, criterion, min_epsilon=0.05, max_epsilon=0.3):
        super(RandomFGSMAttack, self).__init__(model)
        self.criterion = criterion
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon

    def generate(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

        adv_x = torch.clamp(x.detach() + torch.zeros_like(x).uniform_(self.min_epsilon, self.max_epsilon) * torch.sign(x_grad.detach()), 0, 1).detach()

        return adv_x


class TargetedFGSMAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05):
        super(TargetedFGSMAttack, self).__init__(model)
        self.criterion = criterion
        self.epsilon = epsilon

    def generate(self, x, y):
        adv_x = x.detach()
        adv_x.requires_grad = True
        output = self.model(adv_x)
        loss = self.criterion(output, y)

        adv_x_grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

        adv_x = torch.clamp(adv_x - self.epsilon * torch.sign(adv_x_grad), 0, 1).detach()

        return adv_x


class DigitFGSMAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05):
        super(DigitFGSMAttack, self).__init__(model)
        self.criterion = criterion
        self.epsilon = epsilon

    def generate(self, x, y=None):
        x.requires_grad = True
        output = self.model(x)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_grad = torch.autograd.grad(loss, x, only_inputs=True)[0]

        adv_x = torch.clamp(x.detach() + self.epsilon * torch.sign(x_grad.detach()), 0, 1).detach()

        return adv_x * (x != 0).float()


class RFGSMAttack(Attack):
    def __init__(self, model, criterion, alpha=0.025, epsilon=0.05):
        super(RFGSMAttack, self).__init__(model)
        self.criterion = criterion
        self.alpha = alpha
        self.epsilon = epsilon

    def generate(self, x, y=None):
        x_p = torch.clamp(x.detach() + self.alpha * torch.sign(torch.zeros_like(x).float().uniform_(-1,1)), 0, 1)
        x_p.requires_grad = True
        output = self.model(x_p)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_p_grad = torch.autograd.grad(loss, x_p, only_inputs=True)[0]

        adv_x = torch.clamp(x_p.detach() + (self.epsilon - self.alpha) * torch.sign(x_p_grad.detach()), 0, 1).detach()

        return adv_x


class FastAttack(Attack):
    # https://openreview.net/pdf?id=BJx040EFvH
    def __init__(self, model, criterion, epsilon=0.05):
        super().__init__(model)
        self.criterion = criterion
        self.epsilon = epsilon

    def generate(self, x, y=None):
        x_p = torch.clamp(x.detach() + torch.zeros_like(x).float().uniform_(-self.epsilon,self.epsilon), 0, 1)
        x_p.requires_grad = True
        output = self.model(x_p)
        loss = self.criterion(output, output.max(1)[1] if y is None else y)

        x_p_grad = torch.autograd.grad(loss, x_p, only_inputs=True)[0]

        adv_x = torch.clamp(x_p.detach() + self.epsilon * torch.sign(x_p_grad.detach()), 0, 1).detach()

        return adv_x


class MyPGDAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super(MyPGDAttack, self).__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, y=None, maximize=False):
        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if y is None:
            y = self.model(x).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x + self.step_size * torch.sign(grad)
            else:
                grad = normalize_l2(grad)
                adv_x = adv_x + self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        if maximize:
            adv_x = torch.clamp(x + self.epsilon * torch.sign(adv_x - x), 0, 1).detach()

        return adv_x


class StackedMyPGDAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super().__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, y=None, stack_start=0):
        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0,1)

        if y is None:
            y = self.model(x.detach()).max(1)[1].detach()

        total_adv_x = None
        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_x = adv_x.detach() + self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

            if i >= stack_start:
                if total_adv_x is None:
                    total_adv_x = adv_x # adv_x = adv_x.detach()라서 괜찮
                else:
                    total_adv_x = torch.cat((total_adv_x, adv_x), dim=0)

        return total_adv_x


class MyTargetedPGDAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super(MyTargetedPGDAttack, self).__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, y=None, maximize=False):
        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if y is None:
            y = self.model(x).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x - self.step_size * torch.sign(grad)
            else:
                grad = normalize_l2(grad)
                adv_x = adv_x - self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        if maximize:
            adv_x = torch.clamp(x + self.epsilon * torch.sign(adv_x - x), 0, 1).detach()

        return adv_x



class OptimizerAttack(Attack):
    def __init__(self, model, criterion, rate=1e-3, optimizer=torch.optim.Adam, num_steps=20, epsilon=0.05):
        super().__init__(model)
        self.model = model
        self.rate = rate
        self.optimizer = optimizer
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps

    def generate(self, x, y=None):
        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = torch.clamp(adv_x, 0, 1)
        adv_x = torch.autograd.Variable(adv_x, requires_grad=True)

        if y is None:
            y = self.model(x.detach()).max(1)[1].detach()

        input_optimizer = self.optimizer([adv_x], lr=self.rate)
        for i in range(self.num_steps):
            output = self.model(adv_x)
            loss = -self.criterion(output, y)

            input_optimizer.zero_grad()
            loss.backward()
            input_optimizer.step()

            with torch.no_grad():
                adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1)

        return adv_x


class MultiplePGDAttack(Attack):
    # generate에 사이즈 확인 넣기는 오버헤드가 있어서
    # 이건 이미지에 대해서만 쓰는 걸로
    def __init__(self, model, criterion, epsilon=0.3, num_steps=40, step_size=0.01, grad_sign=True):
        super().__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, multiple, y=None):
        if y is None:
            y = self.model(x).max(1)[1].repeat(multiple).detach()
        else:
            y = y.repeat(multiple).detach()

        x = x.repeat(multiple, 1, 1, 1)

        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x + self.step_size * torch.sign(grad)
            else:
                grad = normalize_l2(grad)
                adv_x = adv_x + self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class AllTargetedPGDAttack(Attack):
    def __init__(self, model, criterion, num_classes=10, epsilon=0.3, num_steps=40, step_size=0.01, grad_sign=True):
        super().__init__(model)
        self.model = model
        self.criterion = criterion
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, y=None):
        label_list = []
        for i in range(self.num_classes):
            label_list.append(torch.ones(x.shape[0]) * i)
        y = torch.cat(label_list, dim=0).long().to(x.device)

        x = x.repeat(self.num_classes, 1, 1, 1)

        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1)

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x.detach() - self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_x = adv_x.detach() - self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class MyReversePGDAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super().__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, y=None):
        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0,1)

        if y is None:
            y = self.model(x.detach()).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(-output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x.detach() - self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_x = adv_x.detach() - self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class MultiModelPGDAttack:
    def __init__(self, model_list, criterion, epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        self.model_list = model_list
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x):
        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0,1)

        y_list = []
        for model in self.model_list:
            y = model(x.detach()).max(1)[1].detach()
            y_list.append(y)

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            loss = 0
            for model, y in zip(self.model_list, y_list):
                output = model(adv_x)
                loss += self.criterion(output, y)

            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_x = adv_x.detach() + self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class BestPGDAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.3, num_steps=40, step_size=0.01, max_batch_size=128):
        super().__init__(model)
        self.model = model

        assert criterion.reduction is 'none'
        self.criterion = criterion

        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.idx_tensor = torch.tensor([i for i in range(max_batch_size)])

    def generate(self, x, y=None, include_last=False):
        adv_x = x.clone() + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if y is None:
            y = self.model(x.detach()).max(1)[1].detach()

        total_adv_x = None
        total_loss = None
        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(torch.mean(loss), adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)
            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

            total_adv_x = torch_cat(prev=total_adv_x, next=adv_x.unsqueeze(1), dim=1)
            total_loss = torch_cat(prev=total_loss, next=loss.unsqueeze(1).detach(), dim=1)

        best_indices = torch.max(total_loss, dim=1)[1]
        best_adv_x = total_adv_x[self.idx_tensor[:x.size(0)], best_indices]

        if include_last:
            return best_adv_x, adv_x
        else:
            return best_adv_x


class MultiLossPGDAttack(Attack):
    def __init__(self, model, criterion, epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super().__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, y):
        adv_x = x.clone().detach()
        adv_x += torch.zeros_like(adv_x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0,1)

        pred = self.model(x.detach()).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y) + self.criterion(output, pred)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            if self.grad_sign:
                adv_x = adv_x.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_x = adv_x.detach() + self.step_size * grad

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x

# https://github.com/yaodongyu/TRADES/blob/master/trades.py
# 이게 공식인데 여기서는 loss = self.kl_criterion(F.log_softmax(target, dim=1), F.softmax(output, dim=1)) -> 이건 어디서 보고 이걸 썼었음
# 가 아니라 loss = self.kl_criterion(F.log_softmax(output, dim=1), F.softmax(target, dim=1))을 쓰더라
class TRADESPGDAttack(object):
    def __init__(self, model, kl_criterion=torch.nn.KLDivLoss(reduction='batchmean'),
                 epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super().__init__()
        self.model = model
        self.kl_criterion = kl_criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, target=None):
        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if target is None:
            with torch.no_grad():
                target = self.model(x).detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.kl_criterion(F.log_softmax(output, dim=1), F.softmax(target, dim=1))
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class ReverseTRADESPGDAttack(object):
    def __init__(self, model, kl_criterion=torch.nn.KLDivLoss(reduction='batchmean'),
                 epsilon=0.05, num_steps=20, step_size=0.01, grad_sign=True):
        super().__init__()
        self.model = model
        self.kl_criterion = kl_criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def generate(self, x, target=None):
        adv_x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        adv_x = adv_x.clamp(0, 1).detach()

        if target is None:
            with torch.no_grad():
                target = self.model(x).detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.kl_criterion(F.log_softmax(target, dim=1), F.softmax(output, dim=1))
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class BIMAttack(Attack):
    def __init__(self, model, criterion, epsilon, num_steps, step_size):
        super(BIMAttack, self).__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def generate(self, x, y=None):
        adv_x = x.detach()

        if y is None:
            y = self.model(x).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class MIMAttack(Attack):
    def __init__(self, model, criterion, epsilon, num_steps, step_size, mu=0.):
        super(MIMAttack, self).__init__(model)
        self.model = model
        self.criterion = criterion
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.mu = mu

    def generate(self, x, y=None):
        adv_x = x.detach()
        g = 0.

        if y is None:
            y = self.model(x).max(1)[1].detach()

        for i in range(self.num_steps):
            adv_x.requires_grad = True
            output = self.model(adv_x)
            loss = self.criterion(output, y)
            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]
            grad_shape = grad.shape

            g = self.mu * g + F.normalize(grad.reshape((grad_shape[0], -1)), p=1, dim=1).reshape(grad_shape)

            adv_x = adv_x + self.step_size * torch.sign(g)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x


class CWAttack(Attack):
    def __init__(self, model, k=50, epsilon=8./255., num_steps=7, step_size=2./255.):
        super(CWAttack, self).__init__(model)
        self.model = model
        self.k = k
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size

    def generate(self, x, y=None):
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
            wrong_logit = torch.max(torch.where(onehot_label==1., min_tensor, output), dim=1)

            loss = (-F.relu(correct_logit - wrong_logit + self.k)).sum()

            grad = torch.autograd.grad(loss, adv_x, only_inputs=True)[0]

            adv_x = adv_x + self.step_size * torch.sign(grad)

            adv_x = torch.min(torch.max(adv_x, x - self.epsilon), x + self.epsilon).clamp(0, 1).detach()

        return adv_x

