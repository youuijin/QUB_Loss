import torch
import torch.nn.functional as F
# from attack.AttackBase import Attack

class UAPAttack():
    def __init__(self, model, feature_layer, feature_optim, uaps, feature_layer_idx, momentum, eps=8.0, uap_eps=10.0, mean=(0, 0, 0), std=(1, 1, 1), device=None):
        self.model = model  
        self.feature_layer = feature_layer
        self.device = device
        self.mean = torch.tensor(mean).to(device).view(1, 3, 1, 1)
        self.std = torch.tensor(std).to(device).view(1, 3, 1, 1)

        self.eps = eps/255./self.std
        self.uap_eps = uap_eps/255./self.std
        self.upper_limit = ((1 - self.mean) / self.std)
        self.lower_limit = ((0 - self.mean) / self.std)

        self.uaps = torch.clamp(self.uap_eps * torch.sign(uaps), -self.uap_eps, self.uap_eps)
        self.feature_layer_idx = feature_layer_idx
        self.momentum = momentum
        self.feature_optim = feature_optim # optimizer for feature_layer

    def perturb(self, x_natural, y):
        # self.model.eval()

        delta = torch.zeros_like(x_natural).to(self.device)
        for i, e in enumerate(self.eps.squeeze()):
            delta[:, i, :, :].uniform_(-e, e)

        adv_x = torch.clamp(x_natural + delta, self.lower_limit, self.upper_limit)
        with torch.no_grad():
            feature = self.model(adv_x, feature_layer=self.feature_layer_idx)

        feature, out = self.feature_layer(feature)
        uap_max_idx = feature.max(dim=1)[1]
        self.uap_max_idx = uap_max_idx
        uap_noise = self.uaps[uap_max_idx].clone()

        self.feature_optim.zero_grad()
        loss_uap = F.cross_entropy(out, y)
        loss_uap.backward()
        self.feature_optim.step()

        adv_x = adv_x + uap_noise
        delta = torch.clamp(adv_x - x_natural, -self.eps, self.eps)
        adv_x = torch.clamp(x_natural + delta, self.lower_limit, self.upper_limit).detach()
        adv_x.requires_grad_()

        ori_output = self.model(adv_x)
        ori_loss = F.cross_entropy(ori_output, y)
        ori_loss.backward(retain_graph=True)

        self.grad_x = adv_x.grad.detach()
        adv_x = adv_x + self.eps * adv_x.grad.sign()
        delta = torch.clamp(adv_x - x_natural, -self.eps, self.eps)
        adv_x = torch.clamp(x_natural + delta, self.lower_limit, self.upper_limit).detach()

        # self.model.train()

        return adv_x, ori_output
    
    def update_uap(self):
        grad_norm = torch.norm(self.grad_x, p=1)
        cur_grad = self.grad_x / grad_norm
        for uap_idx in set(self.uap_max_idx.tolist()):
            self.momentum[uap_idx] = cur_grad[uap_idx == self.uap_max_idx].mean(dim=0) + self.momentum[uap_idx] * 0.3
            self.uaps[uap_idx] = torch.clamp(
                self.uaps[uap_idx] + self.uap_eps * torch.sign(self.momentum[uap_idx]), -self.uap_eps, self.uap_eps)

        self.momentum = self.momentum.detach()
        self.uaps = self.uaps.detach()