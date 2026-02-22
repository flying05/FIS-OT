#!/bin/bash
# filepath: /opt/data/private/action_seg_ot/run_desktop.sh

data_path="/opt/data/private/action_seg_ot/data"
gpu=0  # Set to 0

# Run desktop_assembly experiment
'''python3 src/train.py -p $data_path -d desktop_assembly -ac all -c 22 -ne 30 -g $gpu \
    --seed 0 -s --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 \
    -ls 512 128 40 -ua --group desktop_results --wandb -v'''
'''python3 src/train_contrastive_asot1.py -p $data_path -d desktop_assembly -ac all -c 22 -bs 2 -ne 33 -g $gpu \
    --seed 0 -s --rho 0.25 -lat 0.16 -vf 5 -lr 1e-3 -wd 1e-4 -r 0.02 \
    -ls 512 128 40 -ua \
    --contrastive-weight 0.5 --warmup-epochs 3 --group da_contrastive_results --wandb -v'''
python3 src/train_feg_asot.py \
    -p $data_path \
    -d desktop_assembly \
    -ac all \
    -c 22 \
    -bs 2 \
    -ne 34 \
    -g $gpu \
    --seed 0 \
    -s \
    --rho 0.25 \
    -lat 0.16 \
    -vf 5 \
    -lr 1e-3 \
    -wd 1e-4 \
    -r 0.02 \
    -ls 512 128 40 \
    -ua \
    -km \
    --feg-weight 0.5 \
    --feg-warmup 2 \
    --feg-h 0.1 \
    --group da_feg_sota_run \
    --wandb \
    -v
