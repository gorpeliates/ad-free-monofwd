import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from experiments.config import ExperimentConfig
from .trainingutils import (
    early_stopping_improved,
    BPModel,
    build_step_schedulers,
)
from .evaluation import evaluate_bp
from copy import deepcopy
from typing import Optional, Tuple
from log_utils.logging import get_logger

logger = get_logger(__name__)


# region BP Training
def train_bp_one_epoch(
    model: BPModel,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    cfg: ExperimentConfig,
) -> Tuple[float, float]:
    """
    Train any standard backprop model for one epoch.
    Returns:
      total_loss, acc
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = 0

    for x, y in dataloader:
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)
        num_batches += 1

    return total_loss / num_batches, total_correct / total_seen


def run_bp_training(
    model: BPModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
    writer: Optional[SummaryWriter] = None,
) -> dict:
    """Runs standard BP training with early stopping based on validation loss. Returns metrics for BP predictor.
    Returns:
        metrics (dict): Structure is as follows:
        {

            'bp': {'train_losses': [...], 'train_accs': [...], 'val_losses': [...], 'val_accs': [...], 'test_losses': [...], 'test_accs': [...]},
            'early_stopping': {'best_epoch': int, 'stopped_epoch': int}
        }
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    schedulers = build_step_schedulers([optimizer], cfg)
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    best_val_acc = 0.0

    metrics = {
        "bp": {
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
        train_loss, train_acc = train_bp_one_epoch(
            model, optimizer, train_loader, cfg=cfg
        )
        val_loss, val_acc = evaluate_bp(model, val_loader, device=cfg.device)
        model.train()
        best_val_acc = max(best_val_acc, val_acc)

        metrics["bp"]["train_losses"].append(train_loss)
        metrics["bp"]["train_accs"].append(train_acc)
        metrics["bp"]["val_losses"].append(val_loss)
        metrics["bp"]["val_accs"].append(val_acc)

        logger.info(
            f"\n[BP Epoch {epoch}/{cfg.epochs}]\n"
            f"  BP      : train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | best={best_val_acc:.4f}"
        )

        if writer:
            writer.add_scalar("bp/loss/train", train_loss, epoch)
            writer.add_scalar("bp/loss/val", val_loss, epoch)
            writer.add_scalar("bp/acc/train", train_acc, epoch)
            writer.add_scalar("bp/acc/val", val_acc, epoch)

        for scheduler in schedulers:
            scheduler.step()

        if cfg.early_stopping:
            if early_stopping_improved(
                best_val_loss, val_loss, cfg.early_stopping_min_delta
            ):
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = deepcopy(model.state_dict())
                metrics["early_stopping"]["best_epoch"] = epoch
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.early_stopping_patience:
                    metrics["early_stopping"]["stopped_epoch"] = epoch
                    logger.info(
                        f"BP early stopping at epoch {epoch}; best epoch was {metrics['early_stopping']['best_epoch']}."
                    )
                    break

    if cfg.early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = evaluate_bp(model, test_loader, device=cfg.device)
    metrics["bp"]["test_losses"].append(test_loss)
    metrics["bp"]["test_accs"].append(test_acc)
    return metrics


# endregion
