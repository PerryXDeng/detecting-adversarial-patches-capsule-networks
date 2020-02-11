#!/bin/bash
python background_counterfactual_reconstruction.py --load_dir=logs/cifar10/bg_recon_2/20191225_13\:38\:42\:637171_/ --dataset=cifar10 --num_gpus=1 --zeroed_bg_reconstruction=True --batch_size=1
