#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Run experiment for CNNs
echo "Running experiment for CNNs..."
python "$PROJECT_ROOT/src/main.py" --model cnn --dataset all


# Run experiment for MLPs
echo "Running experiment for MLPs..."
python "$PROJECT_ROOT/src/main.py" --model mlp --dataset all
