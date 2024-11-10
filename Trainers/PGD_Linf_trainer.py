from Trainers.trainer_base import Trainer
from attack.pgd_attack import PGDAttack

from datetime import datetime

class PGD_Linf_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'PGD_Linf(eps{args.eps}_alpha{args.alpha}_iter{args.iter}_restart{args.restart})_{self.loss_desc}_lr{args.lr}_{cur}'
        
        # train attack
        self.train_attack = PGDAttack(self.model, eps=args.eps, alpha=args.alpha, iter=args.iter, 
                                      restart=args.restart, mean=self.mean, std=self.std, device=self.device)

    # def train(self): # use parent's method
    # def train_1_epoch(self, _): # use parent's method
    # def valid(self): # use parent's method