#!/bin/sh
# python3 train_val.py --dataset=mnist --batch_size=128 --A=64 --B=8 --C=16 --D=16 --weight_reg=True --affine_voting=False --drop_rate=0.5 --dropout=False --dropconnect=False --recon_loss=True --num_gpus=2 --epoch=30 --logdir=recon_custom_weightreg_no_affine
python3 train_val.py --epoch=500 --num_gpus=2 --lrn_rate=0.00003 --logdir=cnn_low_lr --dataset=mnist --batch_size=32 --cnn=True
# python3 test.py --dataset=mnist --batch_size=2 --num_gpus=1 --load_dir=logs/mnist/recon_custom_weightreg/20191016_13:35:21:577802_/ --ckpt_name=all
