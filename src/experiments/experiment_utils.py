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
import numpy as np
import random

from training.monofwdtrain import run_monofwd_training_ad
from training.ddtrain import run_monofwd_training_dd
from training.bptrain import run_bp_training
from models.mlp.BPMLP import BPMLP

logger = get_logger(__name__)


# architectures per dataset
_MLP_ARCH = {
    "mnist":        (28 * 28,     [500]  * 2),
    "fashionmnist": (28 * 28,     [1000] * 4),
    "cifar10":      (32 * 32 * 3, [2000] * 6),
    "cifar100":     (32 * 32 * 3, [2000] * 6),
}

_CNN_ARCH = {
    "mnist":        [32, 32],
    "fashionmnist": [64, 64],
    "cifar10":      [128, 128],
    "cifar100":     [128, 128],
}


def _build_mono_model(
    cfg: ExperimentConfig, in_channels: int, num_classes: int
) -> nn.Module:
    ds = cfg.dataset.lower()
    if cfg.model == "mlp":
        input_dim, hidden_dims = _MLP_ARCH[ds]
        return MonoFwdMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    if cfg.model == "cnn":
        return MonoFwdCNN(in_ch=in_channels, channels=_CNN_ARCH[ds], num_classes=num_classes)
    raise ValueError(f"Unknown model type: {cfg.model}")


def _build_bp_model(
    cfg: ExperimentConfig, in_channels: int, num_classes: int
) -> nn.Module:
    ds = cfg.dataset.lower()
    if cfg.model == "mlp":
        input_dim, hidden_dims = _MLP_ARCH[ds]
        return BPMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)
    if cfg.model == "cnn":
        return BPCNN(in_ch=in_channels, channels=_CNN_ARCH[ds], num_classes=num_classes, )
    raise ValueError(f"Unknown model type: {cfg.model}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mlp_dimensions(cfg: ExperimentConfig) -> Tuple[int, List[int]]:
    return _MLP_ARCH[cfg.dataset.lower()]


def run_experiment_mlp(cfg: ExperimentConfig, run_name: str) -> dict:
    """
    Trains MLP models according to cfg.train_method:
      - 'all'      : runs autodiff, dd, and backprop; returns keys 'autodiff', 'dd', 'bp'
      - 'autodiff' : returns keys 'mono_ff', 'mono_bp', 'early_stopping'
      - 'dd'       : returns keys 'mono_ff', 'mono_bp', 'early_stopping'
      - 'backprop' : returns keys 'bp', 'early_stopping'
    """
    log_file = setup_logging(cfg, run_name)
    logger.info(f"Logs saved to: {log_file}")

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, in_channels, num_classes = build_dataloaders(
        cfg
    )

    writer = SummaryWriter(log_dir=f"{cfg.tensorboard_logdir}/{run_name}")
    logger.info(f"TensorBoard logs: {cfg.tensorboard_logdir}/{run_name}")

    match cfg.train_method:
        case "autodiff":
            metrics = run_monofwd_training_ad(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader,
                val_loader,
                test_loader,
                cfg,
                writer=writer,
            )
            writer.close()
            return {
                "mono_ff": metrics["mono_ff"],
                "mono_bp": metrics["mono_bp"],
                "early_stopping": metrics["early_stopping"],
            }
        case "dd":
            metrics = run_monofwd_training_dd(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader,
                val_loader,
                test_loader,
                cfg,
                writer=writer,
                tag_prefix="dd"
            )
            writer.close()
            return {
                "mono_ff": metrics["mono_ff"],
                "mono_bp": metrics["mono_bp"],
                "early_stopping": metrics["early_stopping"],
            }
        case "backprop":
            metrics = run_bp_training(
                _build_bp_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader,
                val_loader,
                test_loader,
                cfg,
                writer=writer,
            )
            writer.close()
            return {"bp": metrics["bp"], "early_stopping": metrics["early_stopping"]}
        case "all":
            ad_metrics = run_monofwd_training_ad(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader, val_loader, test_loader, cfg, writer=writer, tag_prefix="autodiff",
            )
            dd_metrics = run_monofwd_training_dd(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader, val_loader, test_loader, cfg, writer=writer, tag_prefix="dd",
            )
            bp_metrics = run_bp_training(
                _build_bp_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader, val_loader, test_loader, cfg, writer=writer, tag_prefix="backprop",
            )
            writer.close()
            return {
                "autodiff": {
                    "mono_ff": ad_metrics["mono_ff"],
                    "mono_bp": ad_metrics["mono_bp"],
                    "early_stopping": ad_metrics["early_stopping"],
                },
                "dd": {
                    "mono_ff": dd_metrics["mono_ff"],
                    "mono_bp": dd_metrics["mono_bp"],
                    "early_stopping": dd_metrics["early_stopping"],
                },
                "bp": {
                    "bp": bp_metrics["bp"],
                    "early_stopping": bp_metrics["early_stopping"],
                },
            }
        case _:
            raise ValueError(f"Unknown train_method: {cfg.train_method}")


def run_experiment_cnn(cfg: ExperimentConfig, run_name: str) -> dict:
    """
    Trains CNN models according to cfg.train_method:
      - 'all'      : runs autodiff, dd, and backprop; returns keys 'autodiff', 'dd', 'bp'
      - 'autodiff' : returns keys 'mono_ff', 'mono_bp', 'early_stopping'
      - 'dd'       : returns keys 'mono_ff', 'mono_bp', 'early_stopping'
      - 'backprop' : returns keys 'bp', 'early_stopping'
    """
    log_file = setup_logging(cfg, run_name)
    logger.info(f"Logs saved to: {log_file}")

    set_seed(cfg.seed)
    train_loader, val_loader, test_loader, in_channels, num_classes = build_dataloaders(
        cfg
    )

    writer = SummaryWriter(log_dir=f"{cfg.tensorboard_logdir}/{run_name}")
    logger.info(f"TensorBoard logs: {cfg.tensorboard_logdir}/{run_name}")

    match cfg.train_method:
        case "autodiff":
            metrics = run_monofwd_training_ad(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader,
                val_loader,
                test_loader,
                cfg,
                writer=writer,
            )
            writer.close()
            return {
                "mono_ff": metrics["mono_ff"],
                "mono_bp": metrics["mono_bp"],
                "early_stopping": metrics["early_stopping"],
            }
        case "dd":
            metrics = run_monofwd_training_dd(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader,
                val_loader,
                test_loader,
                cfg,
                writer=writer,
                tag_prefix="dd"
            )
            writer.close()
            return {
                "mono_ff": metrics["mono_ff"],
                "mono_bp": metrics["mono_bp"],
                "early_stopping": metrics["early_stopping"],
            }
        case "backprop":
            metrics = run_bp_training(
                _build_bp_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader,
                val_loader,
                test_loader,
                cfg,
                writer=writer,
            )
            writer.close()
            return {"bp": metrics["bp"], "early_stopping": metrics["early_stopping"]}
        case "all":
            ad_metrics = run_monofwd_training_ad(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader, val_loader, test_loader, cfg, writer=writer, tag_prefix="autodiff",
            )
            dd_metrics = run_monofwd_training_dd(
                _build_mono_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader, val_loader, test_loader, cfg, writer=writer, tag_prefix="dd",
            )
            bp_metrics = run_bp_training(
                _build_bp_model(cfg, in_channels, num_classes).to(cfg.device),
                train_loader, val_loader, test_loader, cfg, writer=writer, tag_prefix="backprop",
            )
            writer.close()
            return {
                "autodiff": {
                    "mono_ff": ad_metrics["mono_ff"],
                    "mono_bp": ad_metrics["mono_bp"],
                    "early_stopping": ad_metrics["early_stopping"],
                },
                "dd": {
                    "mono_ff": dd_metrics["mono_ff"],
                    "mono_bp": dd_metrics["mono_bp"],
                    "early_stopping": dd_metrics["early_stopping"],
                },
                "bp": {
                    "bp": bp_metrics["bp"],
                    "early_stopping": bp_metrics["early_stopping"],
                },
            }
        case _:
            raise ValueError(f"Unknown train_method: {cfg.train_method}")


def run_experiment(cfg: ExperimentConfig, run_name: str) -> dict:
    """Entry point to run an experiment based on the model type specified in cfg"""
    match cfg.model:
        case "mlp":
            return run_experiment_mlp(cfg, run_name)
        case "cnn":
            return run_experiment_cnn(cfg, run_name)
        case _:
            raise ValueError(f"Unknown model type: {cfg.model}")
