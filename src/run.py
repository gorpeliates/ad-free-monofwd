import argparse
import torch

from experiments.config import ExperimentConfig
from experiments.experiment_utils import run_experiment


import json
from datetime import datetime
from pathlib import Path

DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")


def parse_args() -> list[ExperimentConfig]:
    parser = argparse.ArgumentParser(description="MonoFWD")
    parser.add_argument(
        "--dataset", type=str, nargs="+", default=["mnist"], choices=[*DATASETS, "all"]
    )
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--no-early-stopping",
        action="store_false",
        dest="early_stopping",
        help="Disable early stopping",
    )
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument(
        "--train_method",
        type=str,
        default="all",
        choices=["autodiff", "dd", "backprop", "bp_autodiff", "all"],
        help="Training method to run: autodiff, dd, backprop, bp_autodiff, or all (default)",
    )
    parser.add_argument(
        "--dd_eps", type=float, default=1e-3, help="Perturbation magnitude eps for MF+DD"
    )
    parser.add_argument(
        "--dd_num_perturbations",
        type=int,
        default=1,
        help="Number of perturbation directions P per block per step for MF+DD",
    )
    parser.add_argument(
        "--dd_max_params_per_chunk",
        type=int,
        default=50000,
        help="Max parameters perturbed at once per DD chunk (default: 50000)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["adam", "sgd"],
        help="Optimizer for AD/BP baselines (default: adam). DD always uses raw SGD with lr.",
    )
    parser.add_argument(
        "--cnn_proj_dim",
        type=int,
        default=16,
        help="FFZero channel-wise random projection dimension for CNN (default: 16)",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="runs",
        help="TensorBoard log directory (default: runs/)",
    )
    args = parser.parse_args()
    datasets = (
        list(DATASETS) if "all" in args.dataset else list(dict.fromkeys(args.dataset))
    )

    configs = []
    for dataset in datasets:
        config_args = vars(args).copy()
        config_args["dataset"] = dataset
        config_args["early_stopping"] = config_args.pop("early_stopping", True)
        config_args["tensorboard_logdir"] = config_args.pop("logdir", "runs")
        configs.append(ExperimentConfig(**config_args))

    return configs


if __name__ == "__main__":
    configs = parse_args()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    for config in configs:
        print(f"Running {config.model} on {config.dataset}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = config.run_name(timestamp)

        history = run_experiment(config, run_name)

        history_file = results_dir / f"{run_name}.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        print(f"Training history saved to: {history_file}")
