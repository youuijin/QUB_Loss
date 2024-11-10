from Trainers.trainer_base import Trainer
from attack.fgsm_attack import FGSM_RS_Attack

from datetime import datetime

class FGSM_RS_Trainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        # log_name
        cur = datetime.now().strftime('%m-%d_%H-%M')
        self.log_name = f'FGSM_RS(eps{args.eps}_alpha{args.alpha})_{self.loss_desc}_lr{args.lr}_{cur}'
        
        # train attack
        self.train_attack = FGSM_RS_Attack(self.model, eps=args.eps, alpha=args.alpha, mean=self.mean, std=self.std, device=self.device)

    # def train(self): # use parent's method

    # def train_1_epoch(self, _): # use parent's method
    
    # def valid(self): # use parent's method