import torch

from utils.trainer_utils import set_dataloader, set_model
import torch.optim as optim

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from attack.pgd_attack import PGDAttack

class Trainer:
    def __init__(self, args):
        # paths
        self.log_dir = args.log_dir
        self.save_dir = args.save_dir

        # train environment
        self.device = f'cuda:{args.device}' if args.device>=0 else 'cpu'
        self.epoch = args.epoch
        self.valid_epoch = args.valid_epoch
        self.batch_size = args.batch_size

        # normalize
        self.mean = (0, 0, 0)
        self.std = (1, 1, 1)

        # Dataloader (train & validation)
        self.train_loader, self.valid_loader, self.n_way, self.imgsz = set_dataloader(args.dataset, args.batch_size, self.mean, self.std)

        # Model
        self.model = set_model(model_name=args.model, n_class=self.n_way)
        self.model = self.model.to(self.device)

        # train loss
        self.loss = args.loss
        self.loss_desc = args.loss
        if args.loss == 'QUB' and args.QUB_reg>0:
            self.loss_desc = f'{args.loss}(reg{args.QUB_reg})'
        
        # QUB hyperparameter
        self.K = 0.5
        self.QUB_reg = args.QUB_reg

        # Validation Attack - PGD10
        self.valid_attack = PGDAttack(self.model, eps=args.valid_eps, alpha=2., iter=10, mean=self.mean, std=self.std, device=self.device)

        # optimizer & lr scheduler
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        
        lr_steps = args.epoch * len(self.train_loader)
        if args.scheduler == 'multistep':
            decay_epochs = args.decay_epochs.split(',')
            milestones = [int(lr_steps*int(epoch)/args.epoch) for epoch in decay_epochs]
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        elif args.scheduler == 'cyclic':
            ud_epochs = args.up_and_down_epochs.split(',')
            self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=args.base_lr, max_lr=args.lr, 
                            step_size_up=lr_steps*int(ud_epochs[0])/args.epoch, step_size_down=lr_steps*int(ud_epochs[1])/args.epoch)
        else:
            # no scheduler
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[lr_steps], gamma=1.0)

        # tracking best accuracy 
        self.best_acc, self.best_adv_acc, self.last_acc, self.last_adv_acc = 0, 0, 0, 0

    def cur_QUB_reg(self, epoch):
        return self.QUB_reg*(epoch/self.epoch)

    def train(self):
        # Setting logger
        self.writer = SummaryWriter(f'{self.log_dir}/{self.log_name}')
        
        # train all steps
        for epoch in range(self.epoch):
            train_acc, train_loss = self.train_1_epoch(epoch) # implement in child class trainer

            # logging
            self.writer.add_scalar('train/acc', train_acc, epoch)
            self.writer.add_scalar('train/loss', train_loss, epoch)

            print(f'Epoch {epoch}: accuracy {round(train_acc, 2)}\tloss {round(train_loss, 4)}')

            if epoch%self.valid_epoch == 0:
                valid_acc, valid_adv_acc, valid_loss, valid_adv_loss = self.valid()
                # logging
                self.writer.add_scalar('valid/acc', valid_acc, epoch)
                self.writer.add_scalar('valid/adv_acc', valid_adv_acc, epoch)
                self.writer.add_scalar('valid/loss', valid_loss, epoch)
                self.writer.add_scalar('valid/adv_loss', valid_adv_loss, epoch)

        return self.best_acc, self.best_adv_acc, self.last_acc, self.last_adv_acc

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
        if valid_adv_acc > self.best_adv_acc:
            torch.save(self.model.state_dict(), f'{self.save_dir}/{self.log_name}_best.pt')
            self.best_acc = valid_acc
            self.best_adv_acc = valid_adv_acc

        self.last_acc = valid_acc
        self.last_adv_acc = valid_adv_acc
        torch.save(self.model.state_dict(), f'{self.save_dir}/{self.log_name}_last.pt')

        return valid_acc, valid_adv_acc, valid_loss, valid_adv_loss
