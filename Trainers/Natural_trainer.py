import torch
from datetime import datetime

import torch.nn.functional as F

from Trainers.trainer_base import Trainer

class Natural_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'Natural_{self.loss_desc}_lr{args.lr}_{cur}'

        if args.loss != 'CE':
            raise ValueError("Only CE Loss is available during Naturally Train")
    
    # def train(self): # use parent's method

    def train_1_epoch(self, _):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = F.cross_entropy(outputs, targets)
            
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
    
    def valid(self):
        # valid 1 step
        self.model.eval()
        correct, adv_correct, loss_tot, adv_loss_tot, total = 0, 0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in self.valid_loader:
                # clean sample (SA)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                
                loss_tot += F.cross_entropy(outputs, targets, reduction='sum').item()

                # attacked sample (RA)
                adv_inputs = self.valid_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                _, adv_predicted = adv_outputs.max(1)
                adv_correct += adv_predicted.eq(targets).sum().item()

                adv_loss_tot += F.cross_entropy(adv_outputs, targets, reduction='sum').item()

                total += targets.size(0)
            
        valid_acc = 100.*correct/total
        valid_adv_acc = 100.*adv_correct/total
        valid_loss = loss_tot/total
        valid_adv_loss = adv_loss_tot/total

        # Save checkpoint
        ## different from parent : compare STANDARD accuracy
        if valid_acc > self.best_acc:
            torch.save(self.model.state_dict(), f'{self.save_dir}/{self.log_name}_best.pt')
            self.best_acc = valid_acc
            self.best_adv_acc = valid_adv_acc

        self.last_acc = valid_acc
        self.last_adv_acc = valid_adv_acc
        torch.save(self.model.state_dict(), f'{self.save_dir}/{self.log_name}_last.pt')

        return valid_acc, valid_adv_acc, valid_loss, valid_adv_loss
