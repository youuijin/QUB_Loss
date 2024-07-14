import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class TRADESAttack():
    def __init__(self, model, norm='Linf', eps=8.0, alpha=2.0, iter=10, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)
        
        self.model = model
        self.norm = norm
        self.eps = eps/255./self.std
        self.alpha = alpha/255./self.std
        self.iter = iter
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)
        self.criterion_kl = nn.KLDivLoss(size_average=False)

        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)


    def perturb(self, x_natural, y):
        self.model.eval()

        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(self.device).detach()
        for _ in range(self.iter):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = self.criterion_kl(F.log_softmax(self.model(x_adv), dim=1),
                                    F.softmax(self.model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.eps), x_natural + self.eps)
            x_adv = torch.clamp(x_adv, self.lower_limit, self.upper_limit)

        self.model.train()

        return Variable(x_adv, requires_grad=False)
