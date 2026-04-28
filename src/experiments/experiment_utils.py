
import torch
from typing import Tuple, List
from torch import nn
from models.MonoFwdMLP import *
from experiments.dataset_utils import build_dataloaders
from experiments.config import ExperimentConfig
from experiments.logging_utils import setup_logging, get_logger
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.BPMLP import BPMLP, train_bp_mlp_one_epoch

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
        return MonoFwdMLP(input_dim=input_dim, hidden_dims=hidden_dims, num_classes=num_classes,activation=cfg.activation)

    # TODO
    # if cfg.model == "cnn":
    #     channels = [64, 128, 256, 512]
    #     return MonoForwardCNN(in_ch=in_channels, channels=channels, num_classes=num_classes, use_bn=False)

    raise ValueError(f"Unknown model type: {cfg.model}")

# Optimizer utils
def block_parameter_groups(model: nn.Module) -> List[List[nn.Parameter]]:
    """
    Returns a list of parameter groups, where each group corresponds to the parameters of a single block in the model.
    """
    groups: List[List[nn.Parameter]] = []
    # proj M, and linear weights and biases have the same learning 
    # rate, so we group them together.
    for block in model.blocks:  
        params = list(block.parameters())
        groups.append(params)
    return groups

def build_optimizers(model: nn.Module, cfg: ExperimentConfig) -> List[torch.optim.Optimizer]:
    """Builds a list of optimizers for each block in the model."""
    opts = []
    for params in block_parameter_groups(model):
        opts.append(torch.optim.Adam(params, lr=cfg.lr))
    return opts

# run experiments
def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, pred_mode: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if isinstance(model, MonoFwdMLP):
            logits = model.predict_logits(x, mode=pred_mode)
        else:
            if x.dim() > 2:
                x = x.flatten(1)
            logits = model(x)
            
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen


def run_experiment_mlp(cfg: ExperimentConfig) -> dict:
    """
    Runs the training and evaluation loop comparing MonoFWD and BP models based on the provided configuration.    
    Returns:
        dict: Contains metrics for both 'mono' and 'bp'.
    """
    log_file = setup_logging(cfg)
    logger.info(f"Logs saved to: {log_file}")
    
    set_seed(cfg.seed)
    train_loader, test_loader, in_channels, num_classes = build_dataloaders(cfg)
    
    ds = cfg.dataset.lower()
    if ds in {"mnist", "fashionmnist"}:
        hidden_dims = [1000, 1000]
        input_dim = 28 * 28
    elif ds in {"cifar10", "cifar100"}:
        hidden_dims = [2000, 2000, 2000]
        input_dim = 32 * 32 * 3
    else:
        raise ValueError(f"Unsupported dataset for MLP: {cfg.dataset}")
        
    model_mono = MonoFwdMLP(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        num_classes=num_classes, 
        activation=cfg.activation
    )
    model_bp = BPMLP(
        input_dim=input_dim, 
        hidden_dims=hidden_dims, 
        num_classes=num_classes, 
        activation=cfg.activation
    )
    
    model_mono.to(cfg.device)
    model_bp.to(cfg.device)

    opts_mono = build_optimizers(model_mono, cfg)
    opt_bp = torch.optim.Adam(model_bp.parameters(), lr=cfg.lr)

    best_test_acc_mono = 0.0
    best_test_acc_bp = 0.0
    
    # Track training history
    metrics = {
        'mono': {
            'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': []
        },
        'bp': {
            'train_losses': [], 'train_accs': [], 'test_losses': [], 'test_accs': []
        }
    }

    for epoch in range(1,cfg.epochs+1):
        # train the models
        train_loss_mono, train_acc_mono = train_monofwd_mlp_one_epoch_autodiff(model_mono, opts_mono, train_loader, device=cfg.device, pred_mode=cfg.pred_mode)
        train_loss_bp, train_acc_bp = train_bp_mlp_one_epoch(model_bp, opt_bp, train_loader, device=cfg.device)
        
        # evaluate the models
        test_loss_mono, test_acc_mono = evaluate(model_mono, test_loader, device=cfg.device, pred_mode=cfg.pred_mode)
        test_loss_bp, test_acc_bp = evaluate(model_bp, test_loader, device=cfg.device, pred_mode=cfg.pred_mode)
        
        best_test_acc_mono = max(best_test_acc_mono, test_acc_mono)
        best_test_acc_bp = max(best_test_acc_bp, test_acc_bp)
        
        # Store in history
        metrics['mono']['train_losses'].append(train_loss_mono)
        metrics['mono']['train_accs'].append(train_acc_mono)
        metrics['mono']['test_losses'].append(test_loss_mono)
        metrics['mono']['test_accs'].append(test_acc_mono)
        
        metrics['bp']['train_losses'].append(train_loss_bp)
        metrics['bp']['train_accs'].append(train_acc_bp)
        metrics['bp']['test_losses'].append(test_loss_bp)
        metrics['bp']['test_accs'].append(test_acc_bp)

        logger.info(
            f"[Epoch {epoch}/{cfg.epochs}] "
            f"MONO: train_loss={train_loss_mono:.4f} train_acc={train_acc_mono:.4f} "
            f"test_loss={test_loss_mono:.4f} test_acc={test_acc_mono:.4f} best={best_test_acc_mono:.4f} | "
            f"BP: train_loss={train_loss_bp:.4f} train_acc={train_acc_bp:.4f} "
            f"test_loss={test_loss_bp:.4f} test_acc={test_acc_bp:.4f} best={best_test_acc_bp:.4f}"
        )
    
    return metrics
