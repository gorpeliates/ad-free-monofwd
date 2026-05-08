import torch
from .trainingutils import MonoFwdModel, BPModel
from torch.utils.data import DataLoader
from typing import Tuple
import torch.nn.functional as F


@torch.no_grad()
def evaluate_monofwd(
    model: MonoFwdModel,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, float, float]:
    """
    Evalutes a MonoFwd model on the given dataloader, returning losses and accuracies for both FF and BP predictors.
    """
    model.eval()
    total_loss_ff = 0.0
    total_correct_ff = 0
    total_loss_bp = 0.0
    total_correct_bp = 0
    total_seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits_ff, logits_bp = model.predict_logits(x)
        loss_ff = F.cross_entropy(logits_ff, y)
        loss_bp = F.cross_entropy(logits_bp, y)

        total_loss_ff += float(loss_ff.item()) * x.size(0)
        total_correct_ff += int((logits_ff.argmax(dim=1) == y).sum().item())
        total_loss_bp += float(loss_bp.item()) * x.size(0)
        total_correct_bp += int((logits_bp.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)

    return (
        total_loss_ff / total_seen,
        total_correct_ff / total_seen,
        total_loss_bp / total_seen,
        total_correct_bp / total_seen,
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

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen
