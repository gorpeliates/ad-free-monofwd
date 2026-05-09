#!/bin/bash
#SBATCH --job-name="monofwd_train_agorpelioglu"
#SBATCH --partition=gpu-a100-small    
#SBATCH --time=04:00:00               
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=2
#SBATCH --mem=32G                     
#SBATCH --account=Education-EEMCS-Courses-CSE3000          

module load 2025                      
module load python
module load py-torch         # or py-torch

python src/run.py --logdir ./runs