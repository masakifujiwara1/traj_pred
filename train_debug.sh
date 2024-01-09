# !/bin/bash
echo " Running Training EXP"

CUDA_VISIBLE_DEVICES=0 python3 train.py --lr 0.01 --dataset eth --tag eth_debug --num_epochs 250 && echo "eth Launched." &
