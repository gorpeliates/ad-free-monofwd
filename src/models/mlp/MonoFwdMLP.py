import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
import math


class MonoFwdLinearBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

        n, m = out_dim, num_classes
        self.M = nn.Parameter(torch.empty(n, m))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a = F.relu(self.linear(x))
        g = a @ self.M
        return a, g


class MonoFwdMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int):
        super().__init__()
        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList(
            [MonoFwdLinearBlock(dims[i], dims[i + 1], num_classes) for i in range(len(hidden_dims))]
        )
        self.num_classes = num_classes

    def local_losses_logits(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Returns the local losses for each block.
        The loss for each block is the cross-entropy loss between the goodness scores and the true labels,
        as mentioned in the paper.

        The activations are detached to prevent gradients from flowing back through the previous blocks,
        which allows each block to be trained independently.

        Returns:
            losses: List of local losses for each block.
            logits: List of goodness scores (logits) for each block.

        """

        if x.dim() > 2:
            x = x.flatten(1)

        losses: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []

        h = x
        for block in self.blocks:
            block: MonoFwdLinearBlock
            a, g = block.forward(h)
            losses.append(F.cross_entropy(g, y))
            logits.append(g)

            h = a.detach()

        return losses, logits

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts the logits for the input x using either feedforward (ff) or backpropagation (bp) mode.
        FF mode -> sums the goodness scores from all blocks to make a prediction.
        BP mode -> uses the goodness scores from the last block to make a prediction.
        Returns
            a tuple of FF style prediction and BP style prediction, where each is a tensor of shape (batch_size, num_classes)
        """
        if x.dim() > 2:
            x = x.flatten(1)
        h = x
        all_logits = []
        for block in self.blocks:
            block: MonoFwdLinearBlock
            a, g = block.forward(h)
            all_logits.append(g)
            h = a

        return torch.stack(all_logits, dim=0).sum(dim=0), all_logits[-1]
