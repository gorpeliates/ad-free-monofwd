#!/bin/bash
#SBATCH --job-name="monofwd_train_agorpelioglu"
#SBATCH --partition=gpu-a100-small
#SBATCH --output=testrun.out
#SBATCH --time=04:00:00               
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=16G                     
#SBATCH --account=Education-EEMCS-Courses-CSE3000          

module load 2025                      
module load cuda/12.9     

uv sync
srun uv run src/run.py --logdir runs