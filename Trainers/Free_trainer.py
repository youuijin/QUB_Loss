from Trainers.trainer_base import Trainer
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from datetime import datetime

class Free_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'Free(eps{args.eps}_m{args.m})_{self.loss_desc}_lr{args.lr}_{cur}'

        # train attack
        self.epoch = int(self.epoch/args.m)
        self.m = args.m
        self.eps = args.eps/255.

        self.global_noise = torch.zeros(args.batch_size, 3, self.imgsz, self.imgsz)
        self.global_noise = self.global_noise.to(self.device)
    
    # def train(self): # use parent's method

    def train_1_epoch(self, epoch):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0
        cur_reg = self.cur_QUB_reg(epoch)

        global delta
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            for _ in range(self.m):
                self.optimizer.zero_grad()

                delta = Variable(self.global_noise[0:inputs.size(0)], requires_grad=True).to(self.device)
                advx = torch.clamp(inputs + delta, 0, 1)

                if self.loss == 'CE':
                    adv_outputs = self.model(advx)
                    loss = F.cross_entropy(adv_outputs, targets)  

                elif self.loss == 'QUB':
                    outputs = self.model(inputs)
                    softmax = F.softmax(outputs, dim=1)
                    y_onehot = F.one_hot(targets, num_classes = softmax.shape[1])
                    loss = F.cross_entropy(outputs, targets, reduction='none')
                    
                    adv_outputs = self.model(advx)
                    adv_norm = torch.norm(adv_outputs-outputs, dim=1)

                    upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    
                    if self.QUB_reg>0:
                        adv_CE_loss = F.cross_entropy(adv_outputs, targets, reduction='none')
                        dist = torch.pow(upper_loss-adv_CE_loss, 2)
                        if self.QUB_func=='acc':
                            _, predicted = adv_outputs.max(1)
                            probability = predicted.eq(targets).sum().item()/targets.size(0)
                            # print(probability)
                            cur_reg = self.cur_QUB_reg(epoch, probability)
                        upper_loss += cur_reg*dist
                        # tot_reg += reg_value.sum().item() #TODO: logging

                    loss = upper_loss.mean()

                self.optimizer.zero_grad()
                loss.backward()

                grad = delta.grad
                self.global_noise[0:inputs.size(0)] += (self.eps * torch.sign(grad)).data
                self.global_noise.clamp_(-self.eps, self.eps)

                self.optimizer.step()
                self.scheduler.step()

                loss_tot += loss.item()*targets.size(0)
                _, predicted = adv_outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        train_acc = 100.*correct/total
        train_loss = loss_tot/total

        return train_acc, train_loss
    
    # def valid(self): # use parent's method