model='resnet18'
dataset='cifar10'
lr=0.1
device=0

eps=8.
test_eps=8.

# ## env1
# env=1
# epoch=200

## env2
env=2
epoch=110

# ## env3
# env=3
# epoch=30

ARGS="--model=$model --dataset=$dataset --env=$env --epoch=$epoch --eps=$eps --test_eps=$test_eps --lr=$lr --device=$device"

# natural training
python natural_training.py $ARGS 
python FGSM_CKPT_training.py $ARGS --loss=CE
python FGSM_CKPT_training.py $ARGS --loss=QUB
python FGSM_GA_training.py $ARGS --loss=CE
python FGSM_GA_training.py $ARGS --loss=QUB
python FGSM_PGI_training.py $ARGS --loss=CE
python FGSM_PGI_training.py $ARGS --loss=QUB
python FGSM_RS_training.py $ARGS --loss=CE
python FGSM_RS_training.py $ARGS --loss=QUB
python FGSM_SDI_training.py $ARGS --loss=CE
python FGSM_SDI_training.py $ARGS --loss=QUB
python Free_training.py $ARGS --loss=CE
python Free_training.py $ARGS --loss=QUB
python GAT_training.py $ARGS --loss=CE
python GAT_training.py $ARGS --loss=QUB
python PGD_Linf_training.py $ARGS --loss=CE
python PGD_Linf_training.py $ARGS --loss=QUB

python QUB_training.py $ARGS --loss=CE --init=Z
python QUB_training.py $ARGS --loss=QUB --init=Z
python QUB_training.py $ARGS --loss=CE --init=U
python QUB_training.py $ARGS --loss=QUB --init=U
python QUB_training.py $ARGS --loss=CE --init=B
python QUB_training.py $ARGS --loss=QUB --init=B
python QUB_training.py $ARGS --loss=CE --init=N
python QUB_training.py $ARGS --loss=QUB --init=N