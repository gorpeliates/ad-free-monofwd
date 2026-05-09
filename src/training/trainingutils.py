from typing import List
import torch
from torch import nn
from models.cnn.BPCNN import BPCNN
from models.cnn.MonoFwdCNN import MonoFwdCNN
from models.mlp.BPMLP import BPMLP
from models.mlp.MonoFwdMLP import MonoFwdMLP

from experiments.config import ExperimentConfig

BPModel = BPMLP | BPCNN
MonoFwdModel = MonoFwdMLP | MonoFwdCNN


def block_parameter_groups(model: MonoFwdModel) -> List[List[nn.Parameter]]:
    groups: List[List[nn.Parameter]] = []
    for block in model.blocks:
        groups.append(list(block.parameters()))
    return groups


def build_optimizers(
    model: MonoFwdModel, cfg: ExperimentConfig
) -> List[torch.optim.Optimizer]:
    opts = []
    for params in block_parameter_groups(model):
        opts.append(torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay))
    return opts


def build_plateau_scheduler(
    optimizer: torch.optim.Optimizer, cfg: ExperimentConfig
) -> torch.optim.lr_scheduler.ReduceLROnPlateau | None:
    if not cfg.reduce_lr_on_plateau:
        return None
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.reduce_lr_factor,
        patience=cfg.reduce_lr_patience,
        min_lr=cfg.min_lr,
    )


def early_stopping_improved(
    best_loss: float, current_loss: float, min_delta: float
) -> bool:
    return current_loss < best_loss - min_delta
