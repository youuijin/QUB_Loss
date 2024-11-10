from datetime import datetime
import torch

import torch.nn.functional as F

from Trainers.trainer_base import Trainer
from attack.trades_attack import TRADESAttack

class TRADES_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'TRADES(eps{args.eps}_alpha{args.alpha}_iter{args.iter}_beta{args.beta})_{self.loss_desc}_lr{args.lr}_{cur}'

        # train attack
        self.train_attack = TRADESAttack(self.model, eps=args.eps, alpha=args.alpha, 
                                         iter=args.iter, mean=self.mean, std=self.std, device=self.device)
        self.beta = args.beta
        self.criterion_kl = torch.nn.KLDivLoss(size_average=False)

        if args.loss != 'CE':
            raise ValueError("Only CE Loss is available during TRADES Train")
    
    # def train(self): # use parent's method

    def train_1_epoch(self, _):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            adv_inputs = self.train_attack.perturb(inputs, targets)
            loss_natural = F.cross_entropy(outputs, targets)

            loss_robust = (1.0 / inputs.shape[0]) * self.criterion_kl(F.log_softmax(self.model(adv_inputs), dim=1), F.softmax(self.model(inputs), dim=1)+1e-10)
            loss = loss_natural + self.beta * loss_robust

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