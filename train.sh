# !/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset eth --tag GAT-eth --use_lrschd --num_epochs 250 && echo "eth Launched." &
