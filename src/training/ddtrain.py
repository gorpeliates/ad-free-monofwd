from .trainingutils import (
    MonoFwdModel,
    early_stopping_improved,
)
from .evaluation import evaluate_monofwd
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from experiments.config import ExperimentConfig
from models.cnn.MonoFwdCNN import MonoFwdConvBlock
from models.mlp.MonoFwdMLP import MonoFwdLinearBlock, MonoFwdMLP
from log_utils.logging import get_logger
from typing import Optional

logger = get_logger(__name__)


def _get_pre_proj_activation(block: nn.Module, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a block's feature extraction (conv/linear + norm/bn + activation) without
    the projection M, returning (next_h, pre_proj_a).

    next_h    — the spatially-pooled output fed to the next block (CNN) or the activation itself (MLP)
    pre_proj_a — the activation vector that will be multiplied by M to get logits, shape (B, n).
    """

    if isinstance(block, MonoFwdConvBlock):
        x = block.conv(h)
        x = block.bn(x)
        a = F.relu(x)
        next_h = F.max_pool2d(a, kernel_size=2)
        B, C, H, W = a.shape
        spatial_size = H * W
        if block.A is None or block.A.shape[-1] != C * spatial_size:
            block._init_projection(spatial_size, a.device)
        u = a.reshape(B, C * spatial_size)
        pre_proj_a = u @ block.A.t()  # [B, proj_dim]
    elif isinstance(block, MonoFwdLinearBlock):
        pre_proj_a = F.relu(block.linear(h))
        next_h = pre_proj_a
    else:
        raise TypeError(f"Unsupported block type: {type(block)}")

    return next_h, pre_proj_a


def _get_chunk_indices(
    block: nn.Module,
    w_params: list[nn.Parameter],
    n_W: int,
    max_params_per_chunk: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """
    Returns flat-index tensors defining independent perturbation chunks.

    CNN  -> whole channels are kept together; consecutive channels are grouped
            until their cumulative parameter count reaches max_params_per_chunk.
            Each channel contributes conv.weight slice + conv.bias + bn.weight + bn.bias.
    MLP  -> consecutive non-overlapping chunks of size max_params_per_chunk.
    Projection matrices (M) are always updated as a whole and are not chunked.
    """
    if isinstance(block, MonoFwdConvBlock):
        out_ch = block.conv.out_channels

        flat_offset = 0
        conv_start, n_per_ch = None, None
        scalar_bases: list[int] = []

        for p in w_params:
            n = p.numel()
            if p.shape == block.conv.weight.shape:
                conv_start = flat_offset
                n_per_ch = n // out_ch
            elif n == out_ch:
                scalar_bases.append(flat_offset)
            flat_offset += n

        n_scalar_per_ch = len(scalar_bases)
        params_per_ch = n_per_ch + n_scalar_per_ch
        ch_per_chunk = max(1, max_params_per_chunk // params_per_ch)

        chunks: list[torch.Tensor] = []
        for c_start in range(0, out_ch, ch_per_chunk):
            c_end = min(c_start + ch_per_chunk, out_ch)
            conv_idx = torch.arange(
                conv_start + c_start * n_per_ch,
                conv_start + c_end * n_per_ch,
                device=device,
            )
            scalar_idx = torch.tensor(
                [base + c for c in range(c_start, c_end) for base in scalar_bases],
                device=device,
            )
            chunks.append(torch.cat([conv_idx, scalar_idx]))
        return chunks

    elif isinstance(block, MonoFwdLinearBlock):
        chunk_size = max(1, max_params_per_chunk)
        return [torch.arange(start, min(start + chunk_size, n_W), device=device) for start in range(0, n_W, chunk_size)]

    else:
        raise TypeError(f"Unsupported block type: {type(block)}")


def _get_chunk_row_ranges_M(
    block: nn.Module,
    max_params_per_chunk: int,
) -> list[tuple[int, int]]:
    """
    Row-based chunks for M, analogous to channel-based chunks for conv weights.

    Simple row chunks sized by max_params_per_chunk.
    """
    n_rows, n_cols = block.M.shape
    rows_per_chunk = max(1, max_params_per_chunk // n_cols)
    return [(start, min(start + rows_per_chunk, n_rows)) for start in range(0, n_rows, rows_per_chunk)]


@torch.no_grad()
def train_monofwd_one_epoch_dd(
    model: MonoFwdModel,
    dataloader: DataLoader,
    cfg: ExperimentConfig,
    chunk_indices_W: list[list[torch.Tensor]],
    chunk_row_ranges_M: list[list[tuple[int, int]]],
) -> tuple[float, float, float, float]:
    """
    Train any MonoFwd model for one epoch using directional-derivative updates.
    No autodiff — all gradient estimates come from finite-difference forward passes.

    For each block l and each parameter chunk:
        grad_estimate = (n_chunk/P) · sum_p [( L+ − L- )/(2eps)] · vhat_p
        then a plain SGD step: params -= lr * grad_estimate

    Returns: total_loss_ff, acc_ff, total_loss_bp, acc_bp
    """
    model.to(cfg.device)
    model.train()

    eps = cfg.dd_eps
    P = cfg.dd_num_perturbations
    lr = cfg.lr

    total_loss_ff = 0.0
    total_correct_ff = 0
    total_loss_bp = 0.0
    total_correct_bp = 0
    total_seen = 0
    num_batches = 0
    layer_loss_totals: list[float] = []
    layer_correct_totals: list[int] = []

    for x, y in dataloader:
        x: torch.Tensor
        y: torch.Tensor
        x = x.to(cfg.device, non_blocking=True)
        y = y.to(cfg.device, non_blocking=True)

        # flatten for mlp, keep spatial for cnn
        h = x.flatten(1) if isinstance(model, MonoFwdMLP) else x
        logits_per_layer: list[torch.Tensor] = []

        for i, block in enumerate(model.blocks):
            block: MonoFwdConvBlock | MonoFwdLinearBlock
            # -------- W update --------------------------------
            w_params = [p for name, p in block.named_parameters() if name != "M"]
            n_W = sum(p.numel() for p in w_params)

            # Save BN running stats so perturbation passes don't corrupt them
            bn_state = _save_bn_state(block)

            for chunk_idxs in chunk_indices_W[i]:
                n_chunk = len(chunk_idxs)
                grad_acc_chunk = torch.zeros(n_chunk, device=cfg.device)

                for _ in range(P):
                    v = torch.randn(n_chunk, device=cfg.device)
                    v_hat = v / v.norm()

                    delta = torch.zeros(n_W, device=cfg.device)
                    delta[chunk_idxs] = eps * v_hat

                    # +eps perturbation
                    _apply_perturbation(w_params, delta)
                    _, g_plus = block.forward(h)
                    L_plus = F.cross_entropy(g_plus, y).item()

                    # −2eps (net −eps from baseline)
                    _apply_perturbation(w_params, -2.0 * delta)
                    _, g_minus = block.forward(h)
                    L_minus = F.cross_entropy(g_minus, y).item()

                    # restore weights and BN stats after each pair
                    _apply_perturbation(w_params, delta)
                    _restore_bn_state(block, bn_state)

                    dd = (L_plus - L_minus) / (2.0 * eps)
                    grad_acc_chunk += dd * v_hat

                # Plain SGD step for this chunk
                grad_scaled = (n_chunk / P) * grad_acc_chunk
                delta_update = torch.zeros(n_W, device=cfg.device)
                delta_update[chunk_idxs] = -lr * grad_scaled
                _apply_perturbation(w_params, delta_update)

            # --------------- M update (row-chunked) --------

            next_h, pre_proj_a = _get_pre_proj_activation(block, h)

            for row_start, row_end in chunk_row_ranges_M[i]:
                n_chunk_M = (row_end - row_start) * block.M.shape[1]
                grad_acc_M = torch.zeros(row_end - row_start, block.M.shape[1], device=cfg.device)

                for _ in range(P):
                    u = torch.randn(row_end - row_start, block.M.shape[1], device=cfg.device)
                    u_hat = u / u.norm()

                    block.M.data[row_start:row_end] += eps * u_hat
                    g_plus = pre_proj_a @ block.M
                    L_plus = F.cross_entropy(g_plus, y).item()

                    block.M.data[row_start:row_end] -= 2.0 * eps * u_hat
                    g_minus = pre_proj_a @ block.M
                    L_minus = F.cross_entropy(g_minus, y).item()

                    block.M.data[row_start:row_end] += eps * u_hat  # restore
                    dd = (L_plus - L_minus) / (2.0 * eps)
                    grad_acc_M += dd * u_hat

                block.M.data[row_start:row_end] -= lr * (n_chunk_M / P) * grad_acc_M

            #  collect logits and advance h to next block
            g_final = pre_proj_a @ block.M
            logits_per_layer.append(g_final)
            h = next_h.detach()

        if not layer_loss_totals:
            layer_loss_totals = [0.0] * len(logits_per_layer)
            layer_correct_totals = [0] * len(logits_per_layer)

        final_goodness_ff = torch.stack(logits_per_layer, dim=0).sum(dim=0)
        final_goodness_bp = logits_per_layer[-1]

        for i, g in enumerate(logits_per_layer):
            layer_loss_totals[i] += float(F.cross_entropy(g, y).item())
            layer_correct_totals[i] += int((g.argmax(dim=1) == y).sum().item())

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


def _apply_perturbation(params: list[nn.Parameter], flat_delta: torch.Tensor) -> None:
    """Add a flat delta vector back into a list of parameters in-place."""
    idx = 0
    for p in params:
        n = p.numel()
        p.data += flat_delta[idx : idx + n].view_as(p)
        idx += n


def _save_bn_state(block: nn.Module) -> dict:
    """Snapshot running_mean/running_var of all BN layers in block."""
    state = {}
    for name, m in block.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            state[name] = (m.running_mean.clone(), m.running_var.clone())
    return state


def _restore_bn_state(block: nn.Module, state: dict) -> None:
    """Restore running_mean/running_var saved by _save_bn_state."""
    for name, m in block.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and name in state:
            m.running_mean.copy_(state[name][0])
            m.running_var.copy_(state[name][1])


def run_monofwd_training_dd(
    model: MonoFwdModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
    writer: Optional[SummaryWriter] = None,
    tag_prefix: str = "",
) -> dict:
    """Runs MF+DD training. Same metrics structure as run_monofwd_training."""
    p = f"{tag_prefix}/" if tag_prefix else ""
    best_val_loss = float("inf")
    best_state = None
    bad_epochs = 0
    best_val_acc_ff = 0.0
    best_val_acc_bp = 0.0
    metrics = {
        "mono_ff": {"test_acc": 0.0},
        "mono_bp": {"test_acc": 0.0},
        "early_stopping": {"best_epoch": 0, "stopped_epoch": 0},
    }

    chunk_indices_W: list[list[torch.Tensor]] = []
    chunk_row_ranges_M: list[list[tuple[int, int]]] = []

    for i, block in enumerate(model.blocks):
        w_params = [p for name, p in block.named_parameters() if name != "M"]
        n_W = sum(p.numel() for p in w_params)
        n_M = block.M.numel()
        chunks = _get_chunk_indices(block, w_params, n_W, cfg.dd_max_params_per_chunk, cfg.device)
        m_ranges = _get_chunk_row_ranges_M(block, cfg.dd_max_params_per_chunk)
        chunk_indices_W.append(chunks)
        chunk_row_ranges_M.append(m_ranges)
        kind = "conv/bn" if isinstance(block, MonoFwdConvBlock) else "linear"
        logger.info(
            f"Block {i}: {n_W} {kind} params ({len(chunks)} chunks) + {n_M} projection (M) params ({len(m_ranges)} chunks) = {n_W + n_M} total trainable"
        )

    for epoch in range(1, cfg.epochs + 1):
        (
            train_loss_ff,
            train_acc_ff,
            train_loss_bp,
            train_acc_bp,
            train_layer_losses,
            train_layer_accs,
        ) = train_monofwd_one_epoch_dd(
            model,
            train_loader,
            cfg,
            chunk_indices_W,
            chunk_row_ranges_M,
        )
        val_loss_ff, val_acc_ff, val_loss_bp, val_acc_bp, val_layer_losses, val_layer_accs = evaluate_monofwd(
            model, val_loader, device=cfg.device
        )
        model.train()

        best_val_acc_ff = max(best_val_acc_ff, val_acc_ff)
        best_val_acc_bp = max(best_val_acc_bp, val_acc_bp)

        logger.info(
            f"\n[MF+DD Epoch {epoch}/{cfg.epochs}]\n"
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
                        f"MF+DD early stopping at epoch {epoch}; best epoch was {metrics['early_stopping']['best_epoch']}."
                    )
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    _, test_acc_ff, _, test_acc_bp, _, _ = evaluate_monofwd(model, test_loader, device=cfg.device)
    metrics["mono_ff"]["test_acc"] = test_acc_ff
    metrics["mono_bp"]["test_acc"] = test_acc_bp
    return metrics
