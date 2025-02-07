import torch, random, gc

from Trainers.trainer_base import Trainer
from utils.trainer_utils import *

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

class FGSM_PGI_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # data parallel for FGSM_PGI
        # self.model = torch.nn.DataParallel(self.model, device_ids=[2, 3])

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'FGSM_PGI(eps{args.eps}_alpha{args.alpha}_reset{args.epoch_reset}_mom{args.momentum_decay}_lamb{args.lamb}_new)_{self.loss_desc}_lr{args.lr}_{cur}'
        
        # normalize
        tensor_mean = torch.tensor(self.mean).to(self.device).view(1, 3, 1, 1)
        tensor_std = torch.tensor(self.std).to(self.device).view(1, 3, 1, 1)
        self.upper_limit = ((1 - tensor_mean) / tensor_std)
        self.lower_limit = ((0 - tensor_mean) / tensor_std)

        # train attack
        self.eps = args.eps/255./tensor_std
        self.alpha = args.alpha/255./tensor_std
        self.epoch_reset = args.epoch_reset
        self.momentum_decay = args.momentum_decay
        self.delta_init = 'random'
        self.lamb = args.lamb

        # set dataset
        self.num_of_example = 50000 if args.dataset in ['cifar10', 'cifar100'] else 80000
        train_data, _, self.n_way, self.imgsz = set_dataset(args.dataset, self.mean, self.std)
        train_loader = DataLoader(train_data, batch_size=self.num_of_example, shuffle=True, num_workers=0)
        # self.test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=0)

        for X, y in train_loader:
            # self.Xs, self.Ys = X.to(self.device), y.to(self.device)
            self.Xs, self.Ys = X, y

        self.iter_num = self.num_of_example // args.batch_size + (0 if self.num_of_example % args.batch_size == 0 else 1)

    
    def atta_aug(self, input_tensor, rst):
        batch_size = input_tensor.shape[0]
        x = torch.zeros(batch_size)
        y = torch.zeros(batch_size)
        flip = [False] * batch_size

        for i in range(batch_size):
            flip_t = bool(random.getrandbits(1))
            x_t = random.randint(0, 8)
            y_t = random.randint(0, 8)

            rst[i, :, :, :] = input_tensor[i, :, x_t:x_t + self.imgsz, y_t:y_t + self.imgsz]
            if flip_t:
                rst[i] = torch.flip(rst[i], [2])
            flip[i] = flip_t
            x[i] = x_t
            y[i] = y_t

        return rst, {"crop": {'x': x, 'y': y}, "flipped": flip}

    def train(self):
        self.writer = SummaryWriter(f'{self.log_dir}/{self.log_name}')
        for epoch in range(self.epoch):
            self.model.train()
            correct, loss_tot, total = 0, 0, 0

            cur_reg = self.cur_QUB_reg(epoch)

            batch_size = self.batch_size
            cur_order = np.random.permutation(self.num_of_example)
            batch_idx = -self.batch_size

            if epoch % self.epoch_reset == 0:
                temp = torch.rand(self.num_of_example, 3, self.imgsz, self.imgsz)
                if self.delta_init != 'previous':
                    all_delta = torch.zeros_like(temp).to(self.device)
                    all_momentum=torch.zeros_like(temp).to(self.device)
                if self.delta_init == 'random':
                    for j in range(len(self.eps.squeeze())):
                        all_delta[:, j, :, :].uniform_(-self.eps[0][j][0][0].item(), self.eps[0][j][0][0].item())
                    all_delta.data = torch.clamp(self.alpha * torch.sign(all_delta), -self.eps, self.eps)

            idx = torch.randperm(self.Xs.shape[0])
            
            self.Xs = self.Xs[idx, :,:,:].view(self.Xs.size())
            self.Ys = self.Ys[idx].view(self.Ys.size())
            all_delta=all_delta[idx, :, :, :].view(all_delta.size())
            all_momentum=all_momentum[idx, :, :, :].view(all_delta.size())
            
            for i in range(self.iter_num):
                batch_idx = (batch_idx + batch_size) if batch_idx + batch_size < self.num_of_example else 0
                X=self.Xs[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]].clone().detach()
                y= self.Ys[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]].clone().detach()
                delta = all_delta[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]].clone().detach()
                next_delta = all_delta[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]].clone().detach()

                momentum=all_momentum[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]].clone().detach()
                X, y = X.to(self.device), y.to(self.device)
                batch_size = X.shape[0]

                # ## add Att
                rst = torch.zeros(batch_size, 3, self.imgsz, self.imgsz).to(self.device)
                X, transform_info = self.atta_aug(X, rst)

                delta.requires_grad = True
                ori_output = self.model(X + delta[:X.size(0)])

                ori_loss = F.cross_entropy(ori_output, y)

                decay = self.momentum_decay

                ori_loss.backward(retain_graph=True)
                x_grad = delta.grad.detach()
                grad_norm = torch.norm(x_grad, p=1)
                momentum = x_grad/grad_norm + momentum * decay

                next_delta.data = torch.clamp(delta + self.alpha * torch.sign(momentum), -self.eps, self.eps)
                next_delta.data[:X.size(0)] = torch.clamp(next_delta[:X.size(0)], self.lower_limit - X, self.upper_limit - X)

                delta.data = torch.clamp(delta + self.alpha * torch.sign(x_grad), -self.eps, self.eps)
                delta.data[:X.size(0)] = torch.clamp(delta[:X.size(0)], self.lower_limit - X, self.upper_limit - X)

                delta = delta.detach()

                output = self.model(X + delta[:X.size(0)])

                loss_fn = torch.nn.MSELoss(reduction='mean')
                if self.loss == 'CE':
                    loss = F.cross_entropy(output, y) + self.lamb*loss_fn(output.float(), ori_output.float())
                    # loss = LabelSmoothLoss(output, (label_smoothing).float())+args.lamb*loss_fn(output.float(), ori_output.float())
                elif self.loss == 'QUB':
                    clean_outputs = self.model(X)
                    softmax = F.softmax(clean_outputs, dim=1)
                    y_onehot = F.one_hot(y, num_classes = softmax.shape[1])

                    # adv_inputs = attack.perturb(inputs, targets)
                    # adv_outputs = model(adv_inputs)
                    adv_norm = torch.norm(clean_outputs-output, dim=1)

                    loss = F.cross_entropy(clean_outputs, y, reduction='none')

                    if self.QUB_opt == "QUBAT":
                        lamb = epoch/self.epoch
                        upper_loss = loss + torch.sum((output-clean_outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                        adv_CE_loss = F.cross_entropy(output, y, reduction='none')
                        loss = (1-lamb)*upper_loss + lamb*adv_CE_loss
                        loss = loss.mean()
                    elif self.QUB_opt == "CEQUB":
                        lamb = epoch/self.epoch
                        upper_loss = loss + torch.sum((output-clean_outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                        loss = (1-lamb)*loss + lamb*upper_loss
                        loss = loss.mean()
                    elif self.QUB_opt == "dQUB":
                        upper_loss = 2*loss + torch.sum((output-clean_outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                        loss = upper_loss.mean()
                    else:
                        upper_loss = loss + torch.sum((output-clean_outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                        loss = upper_loss.mean()

                    # upper_loss = loss + torch.sum((output-clean_outputs)*(softmax-y_onehot), dim=1) + self.K/2.0*torch.pow(adv_norm, 2)
                    
                    # if self.QUB_reg>0:
                    #     adv_CE_loss = F.cross_entropy(output, y, reduction='none')
                    #     dist = torch.pow(upper_loss-adv_CE_loss, 2)
                    #     if self.QUB_func=='acc':
                    #         _, predicted = output.max(1)
                    #         probability = predicted.eq(y).sum().item()/y.size(0)
                    #         # print(probability)
                    #         cur_reg = self.cur_QUB_reg(epoch, probability)
                    #     upper_loss += cur_reg*dist
                    #     # tot_reg += reg_value.sum().item() #TODO: logging

                    loss = loss + self.lamb*loss_fn(output.float(), ori_output.float())

                self.optimizer.zero_grad()
                # self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                loss_tot += loss.item() * y.size(0)
                correct += (output.max(1)[1] == y).sum().item()
                total += y.size(0)

                self.scheduler.step()

                all_momentum[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]] = momentum

                all_delta[cur_order[batch_idx:min(self.num_of_example, batch_idx + batch_size)]]=next_delta

                del ori_loss, loss, X, y, rst
                torch.cuda.empty_cache()
                gc.collect()

            # # 메모리 정리
            # torch.cuda.empty_cache()
            # gc.collect()

            train_acc = 100.*correct/total
            train_loss = loss_tot/total

            # logging
            self.writer.add_scalar('train/acc', train_acc, epoch)
            self.writer.add_scalar('train/loss', train_loss, epoch)

            print(f'Epoch {epoch}: accuracy {train_acc}\tloss{train_loss}')

            if epoch%self.valid_epoch == 0:
                valid_acc, valid_adv_acc, valid_loss, valid_adv_loss = self.valid()
                # logging
                self.writer.add_scalar('valid/acc', valid_acc, epoch)
                self.writer.add_scalar('valid/adv_acc', valid_adv_acc, epoch)
                self.writer.add_scalar('valid/loss', valid_loss, epoch)
                self.writer.add_scalar('valid/adv_loss', valid_adv_loss, epoch)

        return self.best_acc, self.best_adv_acc, self.last_acc, self.last_adv_acc

    # def valid(self): # use parent's method