# TRADES
python TRADES_training.py --model=resnet18 --epoch=100 --num_step=10 --beta=5.0 --sche=none --normalize=none --device=0
python TRADES_training.py --model=resnet18 --epoch=100 --num_step=10 --beta=5.0 --sche=multistep --normalize=none --device=0
python TRADES_training.py --model=resnet18 --epoch=200 --num_step=10 --beta=5.0 --sche=none --normalize=none --device=0
python TRADES_training.py --model=resnet18 --epoch=200 --num_step=10 --beta=5.0 --sche=multistep --normalize=none --device=0
python TRADES_training.py --model=wrn_34_10 --epoch=100 --num_step=10 --beta=6.0 --sche=none --normalize=none --device=0
python TRADES_training.py --model=wrn_34_10 --epoch=100 --num_step=10 --beta=6.0 --sche=multistep --normalize=none --device=0
python TRADES_training.py --model=wrn_34_10 --epoch=200 --num_step=10 --beta=6.0 --sche=none --normalize=none --device=0
python TRADES_training.py --model=wrn_34_10 --epoch=200 --num_step=10 --beta=6.0 --sche=multistep --normalize=none --device=0

# Free AT
python Free_training.py --model=wrn_34_10 --epoch=150 --m=8 --normalize=imagenet --device=0
python Free_training.py --model=wrn_34_10 --epoch=200 --m=8 --normalize=imagenet --device=0
