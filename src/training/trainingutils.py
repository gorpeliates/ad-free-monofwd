from typing import List, Optional
import torch
from torch import nn
from models.cnn.BPCNN import BPCNN
from models.cnn.MonoFwdCNN import MonoFwdCNN
from models.mlp.BPMLP import BPMLP
from models.mlp.MonoFwdMLP import MonoFwdMLP

from experiments.config import ExperimentConfig


class LinearWarmupScheduler:
    def __init__(
        self, optimizer: torch.optim.Optimizer, warmup_steps: int, target_lr: float
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.current_step = 0

    def step(self) -> bool:
        if self.current_step < self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            current_lr = self.target_lr * warmup_factor
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = current_lr
            self.current_step += 1
            return False
        return True


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


def build_warmup_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, target_lr: float
) -> Optional[LinearWarmupScheduler]:
    if warmup_steps <= 0:
        return None
    return LinearWarmupScheduler(optimizer, warmup_steps, target_lr)


def early_stopping_improved(
    best_loss: float, current_loss: float, min_delta: float
) -> bool:
    return current_loss < best_loss - min_delta
