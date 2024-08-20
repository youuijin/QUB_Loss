import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Nu_Attack():
    def __init__(self, model, eps=8.0, alpha=4.0, steps=1, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.steps = steps

        self.eps = eps/255./self.std
        self.alpha = alpha/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)
    
    def perturb(self, x_natural, y, alt=1, nuc_reg=4.): 
        outputs = self.model(x_natural)

        x_adv = x_natural + (self.alpha*torch.sign(torch.tensor([0.5]).to(self.device) - torch.rand_like(x_natural)))
        x_adv = torch.clamp(x_adv, self.lower_limit, self.upper_limit)

        y = y.to(self.device)
        eps = self.eps/self.steps
        self.model.eval()

        for _ in range(self.steps):
            x_adv = Variable(x_adv, requires_grad=True)
            adv_outputs = self.model(x_adv)
        
            if alt:
                loss = F.cross_entropy(adv_outputs, y) + nuc_reg*torch.norm(outputs - adv_outputs, 'nuc')/y.shape[0]
            else:
                loss = F.cross_entropy(adv_outputs, y)
            loss.backward()

            per = eps * torch.sign(x_adv.grad.data)
            x_adv = x_adv.data + per.to(self.device)
            x_adv = torch.clamp(x_adv, self.lower_limit, self.upper_limit)

        self.model.train()

        return x_adv
