# natural training
python natural_training.py --model=resnet18 --dataset=cifar10 --epoch=200

# PGD-Linf Adversarial Training
python PGD_Linf_training.py --model=resnet18 --dataset=cifar10 --epoch=200 --num_step=10 --eps=8. --alpha=2.