#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DATASETS=("cifar10" "cifar100" "imagenet" "all")
MODEL=${1:-cnn}

mkdir -p "$PROJECT_ROOT/runs/logs"

echo "Submitting jobs for datasets: ${DATASETS[@]} with model: $MODEL"

for dataset in "${DATASETS[@]}"; do
    echo "Submitting job for dataset: $dataset"
    sbatch --job-name="monofwd_train_${dataset}" \
           --output="$PROJECT_ROOT/runs/logs/train_${dataset}_%j.out" \
           "$PROJECT_ROOT/scripts/train.slurm" "$dataset" "$MODEL"
done

echo "All jobs submitted!" 