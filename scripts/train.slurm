#!/bin/bash
DATASET=${1:-all}
MODEL=${2:-cnn}

#SBATCH --job-name="monofwd_train_${DATASET}"
#SBATCH --partition=gpu-a100-small
#SBATCH --output=runs/logs/train_${DATASET}_%j.out
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

module load 2025
module load cuda/12.9

srun uv run --no-sync python src/run.py --logdir runs --model $MODEL --dataset $DATASET
