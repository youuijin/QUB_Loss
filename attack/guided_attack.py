import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Guided_Attack():
    def __init__(self, model, eps=8.0, alpha=4.0, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.step = 1
        self.l2_reg = 10.0

        self.eps = eps/255./self.std
        self.alpha = alpha/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)
    
    def perturb(self, x_natural, y, alt=1): 
        tar = Variable(y.to(self.device))
        out = self.model(x_natural)
        P_out = torch.nn.Softmax(dim=1)(out)

        eps = self.eps/self.step 
        self.model.eval()

        adv_x = x_natural + (self.alpha*torch.sign(torch.tensor([0.5]).to(self.device) - torch.rand_like(x_natural)))

        for _ in range(self.step):
            adv_x = Variable(adv_x, requires_grad=True)
            out  = self.model(adv_x)
            R_out = torch.nn.Softmax(dim=1)(out)
            loss = F.cross_entropy(out, tar) + alt*self.l2_reg*(((P_out - R_out)**2.0).sum(1)).mean(0) 
            loss.backward()
            per = eps * torch.sign(adv_x.grad.data)
            adv_x = adv_x.data + per.to(self.device) 
            adv_x = torch.clamp(adv_x, self.lower_limit, self.upper_limit)

        delta = adv_x - x_natural
        delta = torch.clamp(delta, -self.eps, self.eps)
        adv_x = x_natural + delta
        adv_x = torch.clamp(adv_x, self.lower_limit, self.upper_limit)

        self.model.train()

        return adv_x