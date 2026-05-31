import torch
from .trainingutils import MonoFwdModel, BPModel
from torch.utils.data import DataLoader
from typing import List, Tuple
import torch.nn.functional as F


@torch.no_grad()
def evaluate_monofwd(
    model: MonoFwdModel,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, float, float, List[float], List[float]]:
    """
    Evalutes a MonoFwd model on the given dataloader, returning losses and accuracies for both FF and BP predictors.
    """
    model.eval()
    total_loss_ff = 0.0
    total_correct_ff = 0
    total_loss_bp = 0.0
    total_correct_bp = 0
    total_seen = 0
    num_batches = 0
    layer_loss_totals: List[float] = []
    layer_correct_totals: List[int] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits_ff, logits_bp = model.predict_logits(x)
        losses, logits_per_layer = model.local_losses_logits(x, y)
        # default reduction='mean' gives average loss per sample in batch
        loss_ff = F.cross_entropy(logits_ff, y)
        loss_bp = F.cross_entropy(logits_bp, y)

        if not layer_loss_totals:
            layer_loss_totals = [0.0] * len(losses)
            layer_correct_totals = [0] * len(losses)

        for i, (loss, logits) in enumerate(zip(losses, logits_per_layer)):
            layer_loss_totals[i] += float(loss.item())
            layer_correct_totals[i] += int((logits.argmax(dim=1) == y).sum().item())

        total_loss_ff += float(loss_ff.item())
        total_correct_ff += int((logits_ff.argmax(dim=1) == y).sum().item())
        total_loss_bp += float(loss_bp.item())
        total_correct_bp += int((logits_bp.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)
        # paper reports average loss per batch, so we count batches instead of samples for averaging
        num_batches += 1

    return (
        total_loss_ff / num_batches,
        total_correct_ff / total_seen,
        total_loss_bp / num_batches,
        total_correct_bp / total_seen,
        [loss / num_batches for loss in layer_loss_totals],
        [c / total_seen for c in layer_correct_totals],
    )


@torch.no_grad()
def evaluate_bp(
    model: BPModel,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    num_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)
        num_batches += 1

    return total_loss / num_batches, total_correct / total_seen
