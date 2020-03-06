#!/bin/sh
# python3 train_val.py --num_gpus=4 --dataset=cifar10 --batch_size=16 --rescap=True --A=128 --B=24 --C=24 --D=32 --E=24 --F=24 --G=32 --epoch=50 --weight_reg=True --nn_weight_reg_lambda=0.00000002 --capsule_weight_reg_lambda=0.00000002 --drop_rate=0.5 --dropout=True --dropconnect=False --logdir rescap
# python3 train_val.py --load_dir=logs/cifar10/bg_recon_0/20200110_01\:47\:59\:339266_/ --epoch=1000 --dataset=cifar10 --num_gpus=2 --capsule_weight_reg_lambda=0.00000002 --batch_size=32 --logdir=bg_recon_0
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir testi
python3 train_val.py --dataset=cifar10 --logdir=new_bg_recon_2 --batch_size=32 --num_gpus=2 --A=256 --B=32 --C=32 --D=32 --E=0 --F=0 --recon_loss_lambda=1.0 --num_bg_classes=2 --X=512 --Y=1024 --multi_weighted_pred_recon=False --weight_reg=True --nn_weight_reg_lambda=2e-08 --capsule_weight_reg_lambda=2e-08 --affine_voting=True --drop_rate=0.5 --dropout=True --dropconnect=false --dropout_extra=True
