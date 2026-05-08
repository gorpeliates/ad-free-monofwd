import torch
from .trainingutils import (
    MonoFwdModel,
    early_stopping_improved,
    build_optimizers,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from experiments.config import ExperimentConfig
from .evaluation import evaluate_monofwd
from log_utils.logging import get_logger
from typing import Tuple, List
from copy import deepcopy

logger = get_logger(__name__)


def train_monofwd_one_epoch_autodiff(
    model: MonoFwdModel,
    optimizers: List[torch.optim.Optimizer],
    dataloader: DataLoader,
    device: str,
) -> Tuple[float, float, float, float]:
    """
    Train any MonoFwd model for one epoch using local autodiff updates per block.
    The model must implement local_losses_logits(x, y) and expose one optimizer per block.
    Returns:
      total_loss_ff, acc_ff, total_loss_bp, acc_bp
    """
    model.to(device)
    model.train()
    total_loss_ff = 0.0
    total_correct_ff = 0
    total_loss_bp = 0.0
    total_correct_bp = 0
    total_seen = 0

    for x, y in dataloader:
        x: torch.Tensor
        y: torch.Tensor

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        losses, logits_per_layer = model.local_losses_logits(x, y)

        for loss in losses:
            loss.backward()

        for opt in optimizers:
            opt.step()

        with torch.no_grad():
            final_goodness_ff = torch.stack(logits_per_layer, dim=0).sum(dim=0)
            final_goodness_bp = logits_per_layer[-1]

            total_loss_ff += float(
                F.cross_entropy(final_goodness_ff, y).item()
            ) * x.size(0)
            total_correct_ff += int((final_goodness_ff.argmax(dim=1) == y).sum().item())

            total_loss_bp += float(
                F.cross_entropy(final_goodness_bp, y).item()
            ) * x.size(0)
            total_correct_bp += int((final_goodness_bp.argmax(dim=1) == y).sum().item())

            total_seen += x.size(0)

    return (
        total_loss_ff / total_seen,
        total_correct_ff / total_seen,
        total_loss_bp / total_seen,
        total_correct_bp / total_seen,
    )


def run_monofwd_training_ad(
    model: MonoFwdModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
) -> dict:
    """Runs MonoFWD training with early stopping based on validation loss. Returns metrics for both FF and BP predictors.
    Returns:
        metrics (dict): Structure is as follows:
        {
            'mono_ff': {'train_losses': [...], 'train_accs': [...], 'val_losses': [...], 'val_accs': [...], 'test_losses': [...], 'test_accs': [...]},
            'mono_bp': {'train_losses': [...], 'train_accs': [...], 'val_losses': [...], 'val_accs': [...], 'test_losses': [...], 'test_accs': [...]},
            'early_stopping': {'best_epoch': int, 'stopped_epoch': int}
        }
    """
    opts = build_optimizers(model, cfg)
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    best_val_acc_ff = 0.0
    best_val_acc_bp = 0.0

    metrics = {
        "mono_ff": {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
            "test_losses": [],
            "test_accs": [],
        },
        "mono_bp": {
            "train_losses": [],
            "train_accs": [],
            "val_losses": [],
            "val_accs": [],
            "test_losses": [],
            "test_accs": [],
        },
        "early_stopping": {"best_epoch": 0, "stopped_epoch": 0},
    }

    for epoch in range(1, cfg.epochs + 1):
        train_loss_ff, train_acc_ff, train_loss_bp, train_acc_bp = (
            train_monofwd_one_epoch_autodiff(
                model, opts, train_loader, device=cfg.device
            )
        )
        val_loss_ff, val_acc_ff, val_loss_bp, val_acc_bp = evaluate_monofwd(
            model, val_loader, device=cfg.device
        )

        best_val_acc_ff = max(best_val_acc_ff, val_acc_ff)
        best_val_acc_bp = max(best_val_acc_bp, val_acc_bp)

        metrics["mono_ff"]["train_losses"].append(train_loss_ff)
        metrics["mono_ff"]["train_accs"].append(train_acc_ff)
        metrics["mono_ff"]["val_losses"].append(val_loss_ff)
        metrics["mono_ff"]["val_accs"].append(val_acc_ff)

        metrics["mono_bp"]["train_losses"].append(train_loss_bp)
        metrics["mono_bp"]["train_accs"].append(train_acc_bp)
        metrics["mono_bp"]["val_losses"].append(val_loss_bp)
        metrics["mono_bp"]["val_accs"].append(val_acc_bp)

        logger.info(
            f"\n[MonoFwd Epoch {epoch}/{cfg.epochs}]\n"
            f"  MONO-FF : train_loss={train_loss_ff:.4f} | train_acc={train_acc_ff:.4f} | val_loss={val_loss_ff:.4f} | val_acc={val_acc_ff:.4f} | best={best_val_acc_ff:.4f}\n"
            f"  MONO-BP : train_loss={train_loss_bp:.4f} | train_acc={train_acc_bp:.4f} | val_loss={val_loss_bp:.4f} | val_acc={val_acc_bp:.4f} | best={best_val_acc_bp:.4f}"
        )

        if cfg.early_stopping_enabled:
            if early_stopping_improved(
                best_val_loss, val_loss_ff, cfg.early_stopping_min_delta
            ):
                best_val_loss = val_loss_ff
                bad_epochs = 0
                best_state = deepcopy(model.state_dict())
                metrics["early_stopping"]["best_epoch"] = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.early_stopping_patience:
                    metrics["early_stopping"]["stopped_epoch"] = epoch
                    logger.info(
                        f"MonoFwd early stopping at epoch {epoch}; best epoch was {metrics['early_stopping']['best_epoch']}."
                    )
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

    test_loss_ff, test_acc_ff, test_loss_bp, test_acc_bp = evaluate_monofwd(
        model, test_loader, device=cfg.device
    )
    metrics["mono_ff"]["test_losses"].append(test_loss_ff)
    metrics["mono_ff"]["test_accs"].append(test_acc_ff)
    metrics["mono_bp"]["test_losses"].append(test_loss_bp)
    metrics["mono_bp"]["test_accs"].append(test_acc_bp)
    return metrics
