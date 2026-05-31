import torch
from .trainingutils import (
    MonoFwdModel,
    early_stopping_improved,
    build_optimizers,
)
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from experiments.config import ExperimentConfig
from .evaluation import evaluate_monofwd
from log_utils.logging import get_logger
from typing import Optional, Tuple, List
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
    Returns: total_loss_ff, acc_ff, total_loss_bp, acc_bp
    """
    model.to(device)
    model.train()
    total_loss_ff = 0.0
    total_correct_ff = 0
    total_loss_bp = 0.0
    total_correct_bp = 0
    total_seen = 0
    num_batches = 0
    layer_loss_totals: List[float] = []
    layer_correct_totals: List[int] = []

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)

        losses, logits_per_layer = model.local_losses_logits(x, y)

        if not layer_loss_totals:
            layer_loss_totals = [0.0] * len(losses)
            layer_correct_totals = [0] * len(losses)

        for loss in losses:
            loss.backward()

        for opt in optimizers:
            opt.step()

        with torch.no_grad():
            for i, (loss, logits) in enumerate(zip(losses, logits_per_layer)):
                layer_loss_totals[i] += float(loss.item())
                layer_correct_totals[i] += int((logits.argmax(dim=1) == y).sum().item())

            final_goodness_ff = torch.stack(logits_per_layer, dim=0).sum(dim=0)
            final_goodness_bp = logits_per_layer[-1]

            total_loss_ff += float(F.cross_entropy(final_goodness_ff, y).item())
            total_correct_ff += int((final_goodness_ff.argmax(dim=1) == y).sum().item())
            total_loss_bp += float(F.cross_entropy(final_goodness_bp, y).item())
            total_correct_bp += int((final_goodness_bp.argmax(dim=1) == y).sum().item())
            total_seen += x.size(0)
            num_batches += 1

    return (
        total_loss_ff / num_batches,
        total_correct_ff / total_seen,
        total_loss_bp / num_batches,
        total_correct_bp / total_seen,
        [loss / num_batches for loss in layer_loss_totals],
        [c / total_seen for c in layer_correct_totals],
    )


def run_monofwd_training_ad(
    model: MonoFwdModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
    writer: Optional[SummaryWriter] = None,
    tag_prefix: str = "",
) -> dict:
    opts = build_optimizers(model, cfg)
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    best_val_acc_ff = 0.0
    best_val_acc_bp = 0.0
    p = f"{tag_prefix}/" if tag_prefix else ""

    metrics = {
        "mono_ff": {"test_acc": 0.0},
        "mono_bp": {"test_acc": 0.0},
        "early_stopping": {"best_epoch": 0, "stopped_epoch": 0},
    }

    for epoch in range(1, cfg.epochs + 1):
        (
            train_loss_ff,
            train_acc_ff,
            train_loss_bp,
            train_acc_bp,
            train_layer_losses,
            train_layer_accs,
        ) = train_monofwd_one_epoch_autodiff(
            model,
            opts,
            train_loader,
            device=cfg.device,
        )
        val_loss_ff, val_acc_ff, val_loss_bp, val_acc_bp, val_layer_losses, val_layer_accs = evaluate_monofwd(
            model, val_loader, device=cfg.device
        )
        model.train()
        best_val_acc_ff = max(best_val_acc_ff, val_acc_ff)
        best_val_acc_bp = max(best_val_acc_bp, val_acc_bp)

        logger.info(
            f"\n[MonoFwd Epoch {epoch}/{cfg.epochs}]\n"
            f"  MONO-FF : train_loss={train_loss_ff:.4f} | train_acc={train_acc_ff:.4f} | val_loss={val_loss_ff:.4f} | val_acc={val_acc_ff:.4f} | best={best_val_acc_ff:.4f}\n"
            f"  MONO-BP : train_loss={train_loss_bp:.4f} | train_acc={train_acc_bp:.4f} | val_loss={val_loss_bp:.4f} | val_acc={val_acc_bp:.4f} | best={best_val_acc_bp:.4f}"
        )

        if writer:
            writer.add_scalar(f"{p}mono_ff/loss/train", train_loss_ff, epoch)
            writer.add_scalar(f"{p}mono_ff/loss/val", val_loss_ff, epoch)
            writer.add_scalar(f"{p}mono_ff/acc/train", train_acc_ff, epoch)
            writer.add_scalar(f"{p}mono_ff/acc/val", val_acc_ff, epoch)
            writer.add_scalar(f"{p}mono_bp/loss/train", train_loss_bp, epoch)
            writer.add_scalar(f"{p}mono_bp/loss/val", val_loss_bp, epoch)
            writer.add_scalar(f"{p}mono_bp/acc/train", train_acc_bp, epoch)
            writer.add_scalar(f"{p}mono_bp/acc/val", val_acc_bp, epoch)
            for i, (layer_loss, layer_acc) in enumerate(zip(train_layer_losses, train_layer_accs)):
                writer.add_scalar(f"{p}layer_loss/layer_{i}/train", layer_loss, epoch)
                writer.add_scalar(f"{p}layer_acc/layer_{i}/train", layer_acc, epoch)
            for i, (layer_loss, layer_acc) in enumerate(zip(val_layer_losses, val_layer_accs)):
                writer.add_scalar(f"{p}layer_loss/layer_{i}/val", layer_loss, epoch)
                writer.add_scalar(f"{p}layer_acc/layer_{i}/val", layer_acc, epoch)

        if cfg.early_stopping:
            if early_stopping_improved(best_val_loss, val_loss_ff, cfg.early_stopping_min_delta):
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

    _, test_acc_ff, _, test_acc_bp, _, _ = evaluate_monofwd(model, test_loader, device=cfg.device)
    metrics["mono_ff"]["test_acc"] = test_acc_ff
    metrics["mono_bp"]["test_acc"] = test_acc_bp
    return metrics
