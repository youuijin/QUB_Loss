from datetime import datetime
import torch

import torch.nn.functional as F

from Trainers.trainer_base import Trainer
from attack.nuclear_attack import Nu_Attack

class NuAT_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'NuAT(eps{args.eps}_alpha{args.alpha}_nuc{args.nuc_reg})_{self.loss_desc}_lr{args.lr}_{cur}'

        # train attack
        self.train_attack = Nu_Attack(self.model, eps=args.eps, alpha=args.alpha, 
                                         mean=self.mean, std=self.std, device=self.device)
        self.nuc_reg = args.nuc_reg

        if args.loss != 'CE':
            raise ValueError("Only CE Loss is available during NuAT Train")
    
    # def train(self): # use parent's method

    def train_1_epoch(self, _):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0

        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss_natural = F.cross_entropy(outputs, targets)

            adv_inputs = self.train_attack.perturb(inputs, targets, alt=i%2, nuc_reg=self.nuc_reg)
            adv_outputs = self.model(adv_inputs)

            reg_loss = torch.norm(outputs - adv_outputs, 'nuc')/inputs.shape[0]
            
            loss = loss_natural + self.nuc_reg*reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_tot += loss.item()*targets.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            self.scheduler.step()

        train_acc = 100.*correct/total
        train_loss = loss_tot/total
        
        return train_acc, train_loss
    
    # def valid(self): # use parent's method