
import torch
from dataclasses import dataclass
from typing import Tuple, List
from torch import nn
from src.models.MonoFwdMLP import MonoFwdMLP

@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "monofwd_mlp"  # monofwd_mlp | monofwd_cnn
    pred_mode: str = "ff"  # ff | bp
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "./data"
    num_workers: int = 2
    val_split: float = 0.1
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "relu"

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
        return MonoFwdMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes)

    # TODO
    # if cfg.model == "cnn":
    #     channels = [64, 128, 256, 512]
    #     return MonoForwardCNN(in_ch=in_channels, channels=channels, num_classes=num_classes, use_bn=False)

    raise ValueError(f"Unknown model type: {cfg.model}")

# Optimizer utils
def block_parameter_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    groups: List[List[nn.Parameter]] = []
    for block in model.blocks:  # type: ignore[attr-defined]
        params = list(block.parameters())
        groups.append(params)
    return groups

def build_optimizers(model: nn.Module, cfg: ExperimentConfig) -> List[torch.optim.Optimizer]:
    opts = []
    for params in block_parameter_groups(model):
        opts.append(torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay))
    return opts

