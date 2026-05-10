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
        opts.append(torch.optim.Adam(params, lr=cfg.lr))
    return opts


def build_step_schedulers(
    optimizers: List[torch.optim.Optimizer], cfg: ExperimentConfig
) -> List[torch.optim.lr_scheduler.LRScheduler]:
    schedulers = []
    for opt in optimizers:
        if cfg.model == "cnn":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=cfg.epochs
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, step_size=cfg.scheduler_step_size, gamma=0.1
            )
        schedulers.append(scheduler)
    return schedulers


def early_stopping_improved(
    best_loss: float, current_loss: float, min_delta: float
) -> bool:
    return current_loss < best_loss - min_delta
