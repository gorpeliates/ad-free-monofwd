import random
import argparse
import numpy as np
import torch

from experiments.config import ExperimentConfig
from experiments.experiment_utils import run_experiment_mlp


import json
from datetime import datetime
from pathlib import Path
    

def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(description="MonoFWD")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist", "cifar10", "cifar100"])
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--pred_mode", type=str, default="ff", choices=["ff", "bp"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="relu", choices=["relu", "tanh"])
    return ExperimentConfig(**vars(parser.parse_args()))

if __name__ == "__main__":
    config = parse_args()
        
    match config.model:
        case "mlp":
            history = run_experiment_mlp(config)
        # case "cnn":
        #     history = run_experiment_cnn(config)
        case _:
            raise ValueError(f"Unknown model type: {config.model}")
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = results_dir / f"{config.model}_{timestamp}.json"
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to: {history_file}")
