import torch
from datetime import datetime

import torch.nn.functional as F

from Trainers.trainer_base import Trainer
from attack.fgsm_attack import FGSM_Attack

class FGSM_GA_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'FGSM_GA(eps{args.eps}_lamb{args.lamb})_{self.loss_desc}_lr{args.lr}_{cur}'

        # train attack
        self.train_attack = FGSM_Attack(self.model, eps=args.eps, mean=self.mean, std=self.std, device=self.device)
        self.lamb = args.lamb

        tensor_mean = torch.tensor(self.mean).to(self.device).view(1, 3, 1, 1)
        tensor_std = torch.tensor(self.std).to(self.device).view(1, 3, 1, 1)
        self.upper_limit = ((1 - tensor_mean) / tensor_std)
        self.lower_limit = ((0 - tensor_mean) / tensor_std)

    def l2_norm_batch(self, v):
        norms = (v ** 2).sum([1, 2, 3]) ** 0.5
        return norms

    # def train(self): # use parent's method

    def train_1_epoch(self, epoch):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0

        cur_reg = self.cur_QUB_reg(epoch)
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.loss == 'CE':
                adv_inputs = self.train_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                loss = F.cross_entropy(adv_outputs, targets)
            elif self.loss == 'QUB':
                outputs = self.model(inputs)
                softmax = F.softmax(outputs, dim=1)
                y_onehot = F.one_hot(targets, num_classes = softmax.shape[1])

                adv_inputs = self.train_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                adv_norm = torch.norm(adv_outputs-outputs, dim=1)

                loss = F.cross_entropy(outputs, targets, reduction='none')

                upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                
                if self.QUB_reg>0:
                    adv_CE_loss = F.cross_entropy(adv_outputs, targets, reduction='none')
                    dist = torch.pow(upper_loss-adv_CE_loss, 2)
                    upper_loss += cur_reg*dist
                    # tot_reg += reg_value.sum().item() #TODO: logging

                loss = upper_loss.mean()

            # Grad Align between Original image & random image
            delta1 = torch.zeros_like(inputs, requires_grad=True).to(self.device)
            output = self.model(inputs + delta1)
            x1_loss = F.cross_entropy(output, targets)
            
            grad1 = torch.autograd.grad(x1_loss, delta1, create_graph=True)[0]

            delta2 = torch.zeros_like(inputs).to(self.device)
            for i, e in enumerate(range(3)):
                delta2[:, i, :, :].uniform_(-e, e)
            delta2 = torch.clamp(delta2, self.lower_limit-inputs, self.upper_limit-inputs)
            delta2.requires_grad = True
            output = self.model(inputs + delta2)
            x2_loss = F.cross_entropy(output, targets)
            grad2 = torch.autograd.grad(x2_loss, delta2, create_graph=True)[0]

            grad1_norms, grad2_norms = self.l2_norm_batch(grad1), self.l2_norm_batch(grad2)
            grad1_normalized = grad1 / grad1_norms[:, None, None, None]
            grad2_normalized = grad2 / grad2_norms[:, None, None, None]
            cos = torch.sum(grad1_normalized * grad2_normalized, (1, 2, 3))
            
            loss += self.lamb * (1.0 - cos.mean())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_tot += loss.item()*targets.size(0)
            _, predicted = adv_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            self.scheduler.step()

        train_acc = 100.*correct/total
        train_loss = loss_tot/total
        
        return train_acc, train_loss
    
    # def valid(self): # use parent's method