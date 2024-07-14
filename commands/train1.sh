
# # FGSM-RS
# python FGSM_RS_training.py --model=preresnet18 --epoch=150 --a2=10.0 --sche=cyclic --normalize=cifar --device=1
# python FGSM_RS_training.py --model=preresnet18 --epoch=200 --a2=10.0 --sche=cyclic --normalize=cifar --device=1
# python FGSM_RS_training.py --model=preresnet18 --epoch=150 --a2=10.0 --sche=multistep --normalize=cifar --device=1
# python FGSM_RS_training.py --model=preresnet18 --epoch=200 --a2=10.0 --sche=multistep --normalize=cifar --device=1

# PGD-Linf Adversarial Training
python PGD_Linf_training.py --model=resnet34 --epoch=150 --num_step=7 --normalize=none --device=1
python PGD_Linf_training.py --model=resnet34 --epoch=200 --num_step=7 --normalize=none --device=1
python PGD_Linf_training.py --model=wrn_34_10 --epoch=150 --num_step=7 --normalize=none --device=1
python PGD_Linf_training.py --model=wrn_34_10 --epoch=200 --num_step=7 --normalize=none --device=1

