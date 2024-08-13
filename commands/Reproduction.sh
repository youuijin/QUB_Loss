# No AT
# python natural_training.py --model=preresnet18 --epoch=150 --sche=multiste

# PGD AT # NOW DOING
python PGD_Linf_training.py --model=resnet34 --epoch=150 --num_steps=7 --sche=multistep --normalize=none --device=3
python PGD_Linf_training.py --model=resnet34 --epoch=200 --num_steps=7 --sche=multistep --normalize=none --device=3
python PGD_Linf_training.py --model=wrn_34_10 --epoch=150 --num_steps=7 --sche=multistep --normalize=none --device=3
python PGD_Linf_training.py --model=wrn_34_10 --epoch=200 --num_steps=7 --sche=multistep --normalize=none --device=3

# TRADES
python TRADES_training.py --model=resnet18 --epoch=100 --num_steps=10 --beta=5.0 --sche=none --normalize=none --device=3
python TRADES_training.py --model=resnet18 --epoch=100 --num_steps=10 --beta=5.0 --sche=multistep --normalize=none --device=3
python TRADES_training.py --model=resnet18 --epoch=200 --num_steps=10 --beta=5.0 --sche=none --normalize=none --device=3
python TRADES_training.py --model=resnet18 --epoch=200 --num_steps=10 --beta=5.0 --sche=multistep --normalize=none --device=3
python TRADES_training.py --model=wrn_34_10 --epoch=100 --num_steps=10 --beta=6.0 --sche=none --normalize=none --device=3
python TRADES_training.py --model=wrn_34_10 --epoch=100 --num_steps=10 --beta=6.0 --sche=multistep --normalize=none --device=3
python TRADES_training.py --model=wrn_34_10 --epoch=200 --num_steps=10 --beta=6.0 --sche=none --normalize=none --device=3
python TRADES_training.py --model=wrn_34_10 --epoch=200 --num_steps=10 --beta=6.0 --sche=multistep --normalize=none --device=3

# Free AT
python Free_training.py --model=wrn_34_10 --epoch=150 --m=8 --sche=multistep --normalize=imagenet --device=3
python Free_training.py --model=wrn_34_10 --epoch=200 --m=8 --sche=multistep --normalize=imagenet --device=3

# FGSM-RS # NOW DOING
python FGSM_RS_training.py --model=preresnet18 --epoch=150 --a2=10.0 --sche=cyclic --normalize=cifar --device=3
python FGSM_RS_training.py --model=preresnet18 --epoch=200 --a2=10.0 --sche=cyclic --normalize=cifar --device=3
python FGSM_RS_training.py --model=preresnet18 --epoch=150 --a2=10.0 --sche=multistep --normalize=cifar --device=3
python FGSM_RS_training.py --model=preresnet18 --epoch=200 --a2=10.0 --sche=multistep --normalize=cifar --device=3

# FGSM-SDI
python FGSM_SDI_training.py --model=resnet18 --epoch=110 --k=20 --normalize=none
python FGSM_SDI_training.py --model=resnet18 --epoch=110 --k=20 --normalize=cifar
python FGSM_SDI_training.py --model=wrn_34_10 --epoch=110 --k=20 --normalize=none
python FGSM_SDI_training.py --model=wrn_34_10 --epoch=110 --k=20 --normalize=cifar

# FGSM-CKPT
python FGSM_CKPT_training.py --model=preresnet18 --epoch=200 --c=2 --normalize=none --device=0
python FGSM_CKPT_training.py --model=preresnet18 --epoch=200 --c=3 --normalize=none --device=0
python FGSM_CKPT_training.py --model=preresnet18 --epoch=200 --c=4 --normalize=none --device=0