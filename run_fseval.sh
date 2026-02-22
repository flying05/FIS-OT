#!/bin/bash

data_path="/opt/data/private/action_seg_ot/data"
gpu=0  # Set to 0

# Run FSeval experiment
python3 src/train_feg_asot.py \
    -p $data_path \
    -d FSeval \
    -ac all \
    -c 12 \
    -bs 4 \
    -ne 30 \
    -g $gpu \
    --seed 0 \
    -s \
    --rho 0.15 \
    -lat 0.11 \
    -vf 5 \
    -lr 1e-3 \
    -wd 1e-4 \
    -ua \
    -km \
    --feg-weight 0.1 \
    --feg-warmup 5 \
    --feg-h 0.5 \
    --group fseval_feg_optimized \
    --wandb \
    -v
