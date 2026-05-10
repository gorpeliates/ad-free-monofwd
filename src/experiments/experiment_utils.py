import torch
from typing import Tuple, List
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from models.cnn.BPCNN import BPCNN
from models.cnn.MonoFwdCNN import MonoFwdCNN
from models.mlp.MonoFwdMLP import MonoFwdMLP
from experiments.dataset_utils import build_dataloaders
from experiments.config import ExperimentConfig
from log_utils.logging import setup_logging, get_logger
from datetime import datetime
import numpy as np
import random

from training.monofwdtrain import run_monofwd_training_ad
from training.ddtrain import run_monofwd_training_dd
from training.bptrain import run_bp_training
from models.mlp.BPMLP import BPMLP

logger = get_logger(__name__)


def build_model(cfg: ExperimentConfig, in_channels: int, num_classes: int) -> nn.Module:
    ds = cfg.dataset.lower()

    if cfg.model == "mlp":
        if ds in {"mnist", "fashionmnist"}:
            hidden_dims = [1000, 1000]
            input_dim = 28 * 28
        elif ds in {"cifar10", "cifar100"}:
            hidden_dims = [2000, 2000, 2000]
            input_dim = 32 * 32 * 3
        else:
            raise ValueError(f"Unsupported dataset for MLP: {cfg.dataset}")
        return MonoFwdMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            num_classes=num_classes,
            activation=cfg.activation,
        )

    if cfg.model == "cnn":
        channels = [64, 128, 256, 512]
        return MonoFwdCNN(
            in_ch=in_channels,
            channels=channels,
            num_classes=num_classes,
        )

    raise ValueError(f"Unknown model type: {cfg.model}")


# run experiments
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp_dimensions(cfg: ExperimentConfig) -> Tuple[int, List[int]]:
    ds = cfg.dataset.lower()
    if ds in {"mnist", "fashionmnist"}:
        return 28 * 28, [1000, 1000]
    if ds in {"cifar10", "cifar100"}:
        return 32 * 32 * 3, [2000, 2000, 2000]
    raise ValueError(f"Unsupported dataset for MLP: {cfg.dataset}")


def run_experiment_mlp(cfg: ExperimentConfig) -> dict:
    """
    Runs the training and evaluation loop comparing MonoFWD and BP models based on the provided configuration.
    Returns:
        dict: Contains metrics for both 'mono' and 'bp'.
    """
    log_file = setup_logging(cfg)
    logger.info(f"Logs saved to: {log_file}")

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, in_channels, num_classes = build_dataloaders(
        cfg
    )
    input_dim, hidden_dims = mlp_dimensions(cfg)

    model_mono = MonoFwdMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        activation=cfg.activation,
    )

    model_bp = BPMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        activation=cfg.activation,
    )

    model_mono.to(cfg.device)
    model_bp.to(cfg.device)

    run_name = f"mlp_{cfg.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{cfg.tensorboard_logdir}/{run_name}")
    logger.info(f"TensorBoard logs: {cfg.tensorboard_logdir}/{run_name}")

    _run_mono = (
        run_monofwd_training_dd
        if cfg.training_method == "dd"
        else run_monofwd_training_ad
    )
    mono_metrics = _run_mono(
        model_mono, train_loader, val_loader, test_loader, cfg, writer=writer
    )
    bp_metrics = run_bp_training(
        model_bp, train_loader, val_loader, test_loader, cfg, writer=writer
    )
    writer.close()

    return {
        "mono_ff": mono_metrics["mono_ff"],
        "mono_bp": mono_metrics["mono_bp"],
        "bp": bp_metrics["bp"],
        "early_stopping": {
            "mono": mono_metrics["early_stopping"],
            "bp": bp_metrics["early_stopping"],
        },
    }


def run_experiment_cnn(cfg: ExperimentConfig) -> dict:
    """
    Runs the training and evaluation loop comparing MonoFWD and BP CNN models.
    Returns:
        dict: Contains metrics for 'mono_ff', 'mono_bp', and 'bp'.
    """
    log_file = setup_logging(cfg)
    logger.info(f"Logs saved to: {log_file}")

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, in_channels, num_classes = build_dataloaders(
        cfg
    )
    channels = [64, 128, 256, 512]

    model_mono = MonoFwdCNN(
        in_ch=in_channels,
        channels=channels,
        num_classes=num_classes,
    )

    model_bp = BPCNN(
        in_ch=in_channels,
        channels=channels,
        num_classes=num_classes,
    )

    model_mono.to(cfg.device)
    model_bp.to(cfg.device)

    run_name = f"cnn_{cfg.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f"{cfg.tensorboard_logdir}/{run_name}")
    logger.info(f"TensorBoard logs: {cfg.tensorboard_logdir}/{run_name}")

    _run_mono = (
        run_monofwd_training_dd
        if cfg.training_method == "dd"
        else run_monofwd_training_ad
    )
    mono_metrics = _run_mono(
        model_mono, train_loader, val_loader, test_loader, cfg, writer=writer
    )
    bp_metrics = run_bp_training(
        model_bp, train_loader, val_loader, test_loader, cfg, writer=writer
    )
    writer.close()

    return {
        "mono_ff": mono_metrics["mono_ff"],
        "mono_bp": mono_metrics["mono_bp"],
        "bp": bp_metrics["bp"],
        "early_stopping": {
            "mono": mono_metrics["early_stopping"],
            "bp": bp_metrics["early_stopping"],
        },
    }


def run_experiment(cfg: ExperimentConfig) -> dict:
    match cfg.model:
        case "mlp":
            return run_experiment_mlp(cfg)
        case "cnn":
            return run_experiment_cnn(cfg)
        case _:
            raise ValueError(f"Unknown model type: {cfg.model}")
