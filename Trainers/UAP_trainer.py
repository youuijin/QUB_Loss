from Trainers.trainer_base import Trainer
from attack.uap_attack import UAPAttack

from datetime import datetime
import torch
import torch.nn.functional as F

class FGSM_UAP_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'FGSM_UAP(eps{args.eps}_uap{args.uap_eps}_lamb{args.lamb}_num{args.uap_num}_new)_{self.loss_desc}_lr{args.lr}_{cur}'
        
        # train attack
        image_shape = (3, self.imgsz, self.imgsz)
        momentum = torch.zeros((args.uap_num, *image_shape)).to(self.device)
        if args.model == 'resnet18' or args.model == 'preresnet18':
            feature_num = 512
            feature_layer_idx = 4
        elif args.model == 'wrn_34_10':
            feature_num = 640
            feature_layer_idx = 3
        feature_layer = FeatureLayer(uap_num=args.uap_num, class_num=self.n_way, feature_num=feature_num).to(self.device)
        opt_feature_layer = torch.optim.Adam(feature_layer.parameters(), lr=0.001)
        uaps = torch.zeros((args.uap_num, *image_shape)).uniform_(-args.uap_eps, args.uap_eps).to(self.device)

        self.train_attack = UAPAttack(self.model, feature_layer, opt_feature_layer, uaps, feature_layer_idx, momentum, eps=args.eps, uap_eps=args.uap_eps, 
                                               mean=self.mean, std=self.std, device=self.device)
        self.lamb = args.lamb
        self.MSE_fn = torch.nn.MSELoss()

    # def train(self): # use parent's method

    def train_1_epoch(self, epoch):
        self.model.train()
        correct, loss_tot, total = 0, 0, 0

        cur_reg = self.cur_QUB_reg(epoch)
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if self.loss == 'CE':
                adv_inputs, ori_adv_output = self.train_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                loss = F.cross_entropy(adv_outputs, targets) + self.lamb * self.MSE_fn(adv_outputs.float(), ori_adv_output.float())
            elif self.loss == 'QUB':
                outputs = self.model(inputs)
                softmax = F.softmax(outputs, dim=1)
                y_onehot = F.one_hot(targets, num_classes = softmax.shape[1])

                adv_inputs, ori_adv_output = self.train_attack.perturb(inputs, targets)
                adv_outputs = self.model(adv_inputs)
                adv_norm = torch.norm(adv_outputs-outputs, dim=1)

                loss = F.cross_entropy(outputs, targets, reduction='none')

                if self.QUB_opt == "QUBAT":
                    lamb = epoch/self.epoch
                    upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    adv_CE_loss = F.cross_entropy(adv_outputs, targets, reduction='none')
                    loss = (1-lamb)*upper_loss + lamb*adv_CE_loss
                    loss = loss.mean()
                elif self.QUB_opt == "CEQUB":
                    lamb = epoch/self.epoch
                    upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    loss = (1-lamb)*loss + lamb*upper_loss
                    loss = loss.mean()
                elif self.QUB_opt == "dQUB":
                    upper_loss = 2*loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    loss = upper_loss.mean()
                else:
                    upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    loss = upper_loss.mean()
                # upper_loss = loss + torch.sum((adv_outputs-outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                
                # if self.QUB_reg>0:
                #     adv_CE_loss = F.cross_entropy(adv_outputs, targets, reduction='none')
                #     dist = torch.pow(upper_loss-adv_CE_loss, 2)
                #     if self.QUB_func=='acc':
                #         _, predicted = adv_outputs.max(1)
                #         probability = predicted.eq(targets).sum().item()/targets.size(0)
                #         # print(probability)
                #         cur_reg = self.cur_QUB_reg(epoch, probability)
                #     upper_loss += cur_reg*dist
                #     # tot_reg += reg_value.sum().item() #TODO: logging

                loss = loss + self.lamb * self.MSE_fn(adv_outputs.float(), ori_adv_output.float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_attack.update_uap()

            loss_tot += loss.item()*targets.size(0)
            _, predicted = adv_outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            self.scheduler.step()

        train_acc = 100.*correct/total
        train_loss = loss_tot/total
        
        return train_acc, train_loss

    # def valid(self): # use parent's method


class FeatureLayer(torch.nn.Module):
    def __init__(self, uap_num=50, class_num=10, feature_num=512):
        super(FeatureLayer, self).__init__()
        self.uap_linear = torch.nn.Linear(feature_num, uap_num)
        self.uap_classifier = torch.nn.Linear(uap_num, class_num)

    def forward(self, x):
        feature = self.uap_linear(x)
        out = self.uap_classifier(F.relu(feature))

        return feature, out
