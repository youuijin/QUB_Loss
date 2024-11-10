from Trainers.trainer_base import Trainer
from attack.fgsm_attack import FGSM_CKPT_Attack

from datetime import datetime

class FGSM_CKPT_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'FGSM_CKPT(eps{args.eps}_alpha{args.alpha}_c{args.c})_{self.loss_desc}_lr{args.lr}_{cur}'
        
        # train attack
        self.train_attack = FGSM_CKPT_Attack(self.model, eps=args.eps, alpha=args.alpha, c=args.c, mean=self.mean, std=self.std, device=self.device)
        
    # def train(self): # use parent's method

    # def train_1_epoch(self, _): # use parent's method
    
    # def valid(self): # use parent's method