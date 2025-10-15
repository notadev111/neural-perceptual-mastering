#!/bin/bash
# Training script for UCL GPU machine

source ~/audio-training-venv/bin/activate
cd ~/neural-perceptual-mastering

python src/train.py \
    --config configs/baseline.yaml \
    --data_dir data/processed \
    --checkpoint_dir checkpoints/exp_001 \
    --log_dir logs/exp_001

echo "Training complete."