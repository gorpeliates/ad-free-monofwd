#!/bin/bash
#SBATCH --partition=gpu-a100-small
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G
#SBATCH --account=Education-EEMCS-Courses-CSE3000

cd "$SLURM_SUBMIT_DIR"

module load 2025
module load cuda/12.9

srun uv run python src/run.py "$@"
