import torch
import torch.nn.functional as F
# from attack.AttackBase import Attack

class FGSM_Attack():
    def __init__(self, model, eps=8.0, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)


    def perturb(self, x_natural, y):
        self.model.eval()

        delta = torch.zeros_like(x_natural).to(self.device)

        delta.requires_grad_()

        output = self.model(x_natural + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.eps*torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)
        delta = delta.detach()

        adv_x = x_natural + delta

        self.model.train()

        return adv_x
    
    def get_grad(self, x_natural, y, uniform=False):
        x = x_natural.clone()

        if uniform:
            delta = torch.zeros_like(x_natural).to(self.device)
            for i, e in enumerate(self.eps.squeeze()):
                delta[:, i, :, :].uniform_(-e, e)
            x = x + delta
            torch.clamp(x, self.lower_limit, self.upper_limit)
            
        x.requires_grad_()

        output = self.model(x)
        
        loss = F.cross_entropy(output, y)
        loss.backward(retain_graph=True)

        grad = x.grad
        
        return grad

class FGSM_RS_Attack():
    def __init__(self, model, eps=8.0, alpha=10.0, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.alpha = alpha/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def perturb(self, x_natural, y):
        self.model.eval()
        # # standarization input

        delta = torch.zeros_like(x_natural).to(self.device)
        # delta = delta.uniform_(-self.a1, self.a1)
        for i, e in enumerate(self.eps.squeeze()):
            delta[:, i, :, :].uniform_(-e, e)

        delta.data = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)

        delta.requires_grad_()

        output = self.model(x_natural + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha*torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)
        delta = delta.detach()

        adv_x = x_natural + delta

        self.model.train()

        return adv_x
    
class Custom_FGSM_RS_Attack():
    def __init__(self, model, eps=8.0, a1=4.0, a2=4.0, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.a1 = a1/255./self.std
        self.a2 = a2/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

    def perturb(self, x_natural, y):
        self.model.eval()
        # # standarization input

        delta = torch.zeros_like(x_natural).to(self.device)
        # delta = delta.uniform_(-self.a1, self.a1)
        for i, e in enumerate(self.a1.squeeze()):
            delta[:, i, :, :].uniform_(-e, e)

        delta.data = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)

        delta.requires_grad_()

        output = self.model(x_natural + delta)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.a2*torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)
        delta = delta.detach()

        adv_x = x_natural + delta

        self.model.train()

        return adv_x


class FGSM_CKPT_Attack():
    def __init__(self, model, eps=8.0, alpha=10.0, c=3, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.device = device
        self.c = c
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.alpha = alpha/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)


    def perturb(self, x_natural, y):
        batch_size = x_natural.shape[0]

        self.model.eval()
        outputs = self.model(x_natural)

        delta = torch.zeros_like(x_natural).to(self.device)
        # delta = delta.uniform_(-self.a1, self.a1)
        for i, e in enumerate(self.eps.squeeze()):
            delta[:, i, :, :].uniform_(-e, e)

        delta.data = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)

        delta.requires_grad_()

        adv_outputs = self.model(x_natural + delta)
        loss = F.cross_entropy(adv_outputs, y)
        loss.backward()

        grad = delta.grad.detach()
        delta.data = torch.clamp(delta + self.alpha*torch.sign(grad), -self.eps, self.eps)

        delta = torch.clamp(delta, self.lower_limit - x_natural, self.upper_limit - x_natural)
        delta = delta.detach()

        adv_x = x_natural + delta

        # Get correctly classified indexes.
        _, pre_clean = torch.max(outputs.data, 1)
        correct = (pre_clean == y)
        correct_idx = torch.masked_select(torch.arange(batch_size).to(self.device), correct)
        wrong_idx = torch.masked_select(torch.arange(batch_size).to(self.device), ~correct)
        
        # Use misclassified images as final images.
        adv_x[wrong_idx] = x_natural[wrong_idx].detach()
        
        # Make checkpoints.
        # e.g., (batch_size*(c-1))*3*32*32 for CIFAR10.
        Xs = (torch.cat([x_natural]*(self.c-1)) + \
              torch.cat([torch.arange(1, self.c).to(self.device).view(-1, 1)]*batch_size, dim=1).view(-1, 1, 1, 1) * \
              torch.cat([delta/self.c]*(self.c-1)))
        Ys = torch.cat([y]*(self.c-1))

        # Inference checkpoints for correct images.
        idx = correct_idx
        idxs = []
        self.model.eval()
        with torch.no_grad():
            for k in range(self.c-1):
                # Stop iterations if all checkpoints are correctly classiffied.
                if len(idx) == 0:
                    break
                # Stack checkpoints for inference.
                elif (batch_size >= (len(idxs)+1)*len(idx)):
                    idxs.append(idx + k*batch_size)
                else:
                    pass
                
                # Do inference.
                if (batch_size < (len(idxs)+1)*len(idx)) or (k==self.c-2):
                    # Inference selected checkpoints.
                    idxs = torch.cat(idxs).to(self.device)
                    pre = self.model(Xs[idxs]).detach()
                    _, pre = torch.max(pre.data, 1)
                    correct = (pre == Ys[idxs]).view(-1, len(idx))
                    
                    # Get index of misclassified images for selected checkpoints.
                    max_idx = idxs.max() + 1
                    wrong_idxs = (idxs.view(-1, len(idx))*(1-correct*1)) + (max_idx*(correct*1))
                    wrong_idx, _ = wrong_idxs.min(dim=0)
                    
                    wrong_idx = torch.masked_select(wrong_idx, wrong_idx < max_idx)
                    update_idx = wrong_idx%batch_size
                    adv_x[update_idx] = Xs[wrong_idx]
                    
                    # Set new indexes by eliminating updated indexes.
                    idx = torch.tensor(list(set(idx.cpu().data.numpy().tolist())\
                                            -set(update_idx.cpu().data.numpy().tolist())))
                    idxs = []
    
        self.model.train()
        return adv_x.detach().to(self.device)