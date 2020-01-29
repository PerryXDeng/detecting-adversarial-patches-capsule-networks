#!/bin/sh
python3 train_val.py --num_gpus=4 --dataset=imagenet64 --batch_size=16 --rescap=True --A=512 --B=64 --C=128 --D=128 --epoch=200 --weight_reg=True --nn_weight_reg_lambda=0.0000002 --capsule_weight_reg_lambda=0.0000002 --drop_rate=0.5 --dropout=True --dropout_extra=True --dropconnect=False --recon_loss=True --num_bg_classes=2 --X=1024 --Y=2048 --logdir=512_64_128_128_recon2bg_1024_2048
# python3 train_val.py --num_gpus=4 --dataset=imagenet64 --batch_size=16 --rescap=True --A=512 --B=64 --C=128 --D=128 --E=128 --epoch=200 --weight_reg=True --nn_weight_reg_lambda=0.0000002 --capsule_weight_reg_lambda=0.0000002 --drop_rate=0.5 --dropout=True --dropout_extra=True --dropconnect=False --recon_loss=True --num_bg_classes=2 --X=1024 --Y=2048 --logdir=512_64_128_128_recon2bg_1024_2048
# python3 train_val.py --num_gpus=4 --dataset=imagenet64 --batch_size=16 --rescap=True --A=512 --B=64 --C=128 --D=128 --E=128 --F=128 --epoch=200 --weight_reg=True --nn_weight_reg_lambda=0.0000002 --capsule_weight_reg_lambda=0.0000002 --drop_rate=0.5 --dropout=True --dropout_extra=True --dropconnect=False --recon_loss=True --num_bg_classes=2 --X=1024 --Y=2048 --logdir=512_64_128_128_recon2bg_1024_2048
# python3 train_val.py --batch_size 8 --num_gpus 1 --epoch 1 --logdir test