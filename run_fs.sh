#!/bin/bash
# filepath: /opt/data/private/action_seg_ot/run_fs.sh

data_path="/opt/data/private/action_seg_ot/data"
gpu=0  # Set to 0

# Run FS experiment
#python3 src/train.py -p $data_path -d FS -ac all -c 19 -ne 30 -g $gpu \
    #--seed 42 -s --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua --group fs_results --wandb -v
#python3 src/train_contrastive_asot1.py -p $data_path -d FS -ac all -c 19 -bs 1 -ne 30 -g $gpu \
    #--seed 0 -s --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua \
    #--contrastive-weight 0.9 --warmup-epochs 5 --group fs_contrastive_results --wandb -v
'''python3 src/train_feg_asot.py \
    -p $data_path -d FS -ac all -c 19 \
    -bs 2 \
    -ne 50 \
    -g $gpu \
    --seed 0 -s --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua \
    -km \
    --feg-weight 0.4 --feg-warmup 10 --feg-h 0.1 \
    --group fs_feg_optimized --wandb -v'''
'''python3 src/train_feg_asot.py \
    -p $data_path -d FS -ac all \
    -c 19 \
    -bs 4 \
    -ne 50 \
    -g $gpu \
    --seed 0 -s --rho 0.15 -lat 0.15 -vf 5 -lr 1e-3 -wd 1e-4 -ua \
    -km \
    --feg-weight 0.2 --feg-warmup 5 --feg-h 0.05 \
    --group fs_feg_bs4_refined --wandb -v'''

python3 src/train_feg_asot.py \
    -p $data_path -d FS -ac all \
    -c 19 \
    -bs 4\
    -ne 60 \
    -g $gpu \
    --seed 0 -s --rho 0.15 -lat 0.15 -vf 5  -lr 1e-3 -wd 1e-4 -ua \
    -km \
    --feg-weight 0.55 --feg-warmup 7 --feg-h 0.1 \
    --group fs_feg_hybrid_best --wandb -v
