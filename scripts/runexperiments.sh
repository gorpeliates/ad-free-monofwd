#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run experiment for CNNs
echo "Running experiment for CNNs..."
python "$PROJECT_ROOT/src/run.py" --model cnn --dataset all


# Run experiment for MLPs
echo "Running experiment for MLPs..."
python "$PROJECT_ROOT/src/run.py" --model mlp --dataset all

echo "Running experiment for MLPs with DD training..."
python src/run.py --model mlp --training_method dd --dd_num_perturbations 2

echo "Running experiment for CNNs with DD training..."
python src/run.py --model cnn --training_method dd --dd_num_perturbations 2