#!/bin/bash

actions=("pancake" "salat" "friedegg" "scrambledegg" "sandwich" "juice" "milk" "tea" "cereals" "coffee")
clusters=(14 8 9 12 9 8 5 7 5 7)
seed=0
gpu=0 # original code is 0
data_path="/opt/data/private/action_seg_ot/data"  # Modify to your actual data path
# --- New: Log directory ---
log_dir="bf_logs"
mkdir -p $log_dir
'''for i in ${!actions[@]}; do
	python3 src/train.py -p $data_path -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 15 -g $gpu --seed 0 -s --rho 0.2 -lat 0.1 -r 0.04 -ae 0.7 -at 0.4 -lr 1e-3 -wd 1e-4 -vf 5 --group main_results --wandb -v -ua 2>&1 | tee "$log_dir/${actions[$i]}.log"
done'''
# Added -p $data_path

'''for i in ${!actions[@]}; do
	python3 src/train_contrastive_asot1.py -p $data_path -d Breakfast -ac ${actions[$i]} -c ${clusters[$i]} -ne 15 -g $gpu --seed 0 -s --rho 0.2 -lat 0.1 -r 0.04 -ae 0.7 -at 0.4 -lr 1e-3 -wd 1e-4 -vf 5 --contrastive-weight 0.3 \
        --warmup-epoch 5 \
		--group bf_contrastive_${actions[$i]} --wandb -v -ua 2>&1 | tee "$log_dir/${actions[$i]}.log"
done'''
# Added -p $data_path
for i in ${!actions[@]}; do
    python3 src/train_feg_asot.py \
        -p $data_path \
        -d Breakfast \
        -ac ${actions[$i]} \
        -c ${clusters[$i]} \
        -ne 20 \
        -g $gpu \
        --seed 0 \
        -s \
        -bs 8 \
        --rho 0.2 \
        -lat 0.1 \
        -r 0.04 \
        -ae 0.7 \
        -at 0.4 \
        -lr 1e-3 \
        -wd 1e-4 \
        -vf 5 \
        -ua \
        -km \
        --feg-weight 0.3 \
        --feg-warmup 5 \
        --feg-h 0.1 \
        --group bf_feg_${actions[$i]} \
        --wandb \
        -v 2>&1 | tee "$log_dir/${actions[$i]}.log"
done
python3 calc_bf_auto.py
