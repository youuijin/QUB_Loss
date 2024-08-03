import torch
import torch.nn.functional as F

class Quadratic_Attack():
    def __init__(self, model, eps=8.0, alpha=4.0, init='uniform', mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.alpha = alpha/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

        if init == 'Z':
            self.get_dist = self.get_standard_fgsm
        elif init == 'U':
            self.get_dist = self.get_uniform_distribution
        elif init == 'B':
            self.get_dist = self.get_bernoulli_distribution
        elif init == 'N':
            self.get_dist = self.get_normal_distribution

    
    def perturb(self, x_natural, y): 
        self.model.eval()

        delta = torch.zeros_like(x_natural).to(self.device)
        delta = self.get_dist(delta)
        # delta = self.alpha*torch.sign(torch.tensor([0.5]).to(self.device) - torch.rand_like(x_natural))
        delta.requires_grad_()

        outputs = self.model(x_natural)
        adv_outputs = self.model(x_natural + delta)
        y_onehot = F.one_hot(y, num_classes = softmax.shape[1])

        softmax = F.softmax(outputs, dim=1)

        adv_norm = torch.norm(adv_outputs-outputs, dim=1)

        loss = F.cross_entropy(x_natural, y, reduction='none')

        upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + 0.5/2.0*torch.pow(adv_norm, 2)
        loss = upper_loss.mean()

        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.eps*torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)
        delta = delta.detach()

        adv_x = x_natural + delta

        self.model.train()

        return adv_x
    
    def get_standard_fgsm(self, a):
        return a

    def get_uniform_distribution(self, a):
        # FGSM-RS (Fast is better than free: Revisiting adversarial training, arXiv 2020)
        ## Initialize: eps * uniform(-1, 1)
        a = a.uniform_(-self.alpha, self.alpha)
        return a

    def get_bernoulli_distribution(self, a):
        # FGSM-BR (Guided adversarial attack for evaluating and enhancing adversarial defenses, NeurIPS 2020)
        ## Initialize: eps/2 * Bernoulli(-1, 1)
        a = self.alpha * a.bernoulli_(p=0.5)
        return a

    def get_normal_distribution(self, a):
        # FGSM-NR (Ensemble adversarial training: Attacks and defenses, ICLR 2018)
        ## Initialze: eps/2 * Normal(0, 1)
        a = self.alpha * a.normal_(0, 1)
        return a