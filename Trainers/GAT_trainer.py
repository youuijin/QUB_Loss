from datetime import datetime
import torch

import torch.nn.functional as F

from Trainers.trainer_base import Trainer
from attack.guided_attack import Guided_Attack

class GAT_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'GAT(eps{args.eps}_alpha{args.alpha}_lamb{args.lamb})_{self.loss_desc}_lr{args.lr}_{cur}'

        # train attack
        self.train_attack = Guided_Attack(self.model, eps=args.eps, alpha=args.alpha, device=self.device)
        self.lamb = args.lamb
        self.reg_mul = args.reg_mul

        if args.loss != 'CE':
            raise ValueError("Only CE Loss is available during GAT Train")
    
    # def train(self): # use parent's method

    def train_1_epoch(self, epoch):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0

        if epoch == int(self.epoch*0.85):
            self.lamb *= self.reg_mul
        
        for i, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # default setting  
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
            adv_inputs = self.train_attack.perturb(inputs, targets, alt=i%2)
            adv_outputs = self.model(adv_inputs)

            Q_out = torch.nn.Softmax(dim=1)(adv_outputs)
            P_out = torch.nn.Softmax(dim=1)(outputs)

            reg_loss = ((P_out - Q_out)**2.0).sum(1).mean(0)

            loss = loss + self.lamb * reg_loss

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