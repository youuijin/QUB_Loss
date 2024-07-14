python FGSM_RS_training.py --model=preresnet18 --epoch=150 --a2=10.0 --sche=cyclic --normalize=cifar --device=3
# python FGSM_RS_training.py --model=preresnet18 --epoch=200 --a2=10.0 --sche=cyclic --normalize=cifar --device=3
python FGSM_RS_training.py --model=preresnet18 --epoch=150 --a2=10.0 --sche=multistep --normalize=cifar --device=3
# python FGSM_RS_training.py --model=preresnet18 --epoch=200 --a2=10.0 --sche=multistep --normalize=cifar --device=3
