#!/bin/bash
#SBATCH --job-name="pytorch/howmanygpus"
#SBATCH --output=howmanygpus.out
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu-a100-small
#SBATCH --mem-per-cpu=1G
#SBATCH --account=Education-EEMCS-Courses-CSE3000         

module load 2025
module load openmpi
module load py-torch

srun python howmanygpus.py