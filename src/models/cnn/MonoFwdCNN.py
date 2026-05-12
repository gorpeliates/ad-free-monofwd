import torch
import torch.nn as nn
from typing import Tuple, List
import torch.nn.functional as F
import math


class MonoFwdConvBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # FFzero CNN: kernel 6x6, stride 1, padding 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=6, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.dropout_p = dropout

        m = num_classes
        n = out_ch
        self.M = nn.Parameter(torch.empty(n, m))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conv -> BN -> ReLU -> MaxPool (-> Dropout), per FFzero Supp. Fig. 4.
        Returns (next_h, goodness).
        """
        x = self.conv(x)
        x = self.bn(x)
        a = F.relu(x)
        next_h = F.max_pool2d(a, kernel_size=2, stride=2)
        if self.dropout_p > 0.0 and self.training:
            next_h = F.dropout(next_h, p=self.dropout_p)

        pooled_a = F.adaptive_avg_pool2d(a, (1, 1)).flatten(1)
        g = pooled_a @ self.M

        return next_h, g


class MonoFwdCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        num_classes: int,
        conv_dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        dims = [in_ch] + channels
        self.blocks = nn.ModuleList(
            [
                MonoFwdConvBlock(
                    dims[i], dims[i + 1], num_classes, dropout=conv_dropout
                )
                for i in range(len(channels))
            ]
        )

    def local_losses_logits(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

        losses: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []

        h = x
        for block in self.blocks:
            block: MonoFwdConvBlock
            a, g = block.forward(h)
            losses.append(F.cross_entropy(g, y))
            logits.append(g)
            h = a.detach()

        return losses, logits

    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FF mode sums goodness from all blocks; BP mode uses the last block.
        """
        h = x
        all_goodness = []
        for block in self.blocks:
            block: MonoFwdConvBlock
            a, g = block.forward(h)
            all_goodness.append(g)
            h = a.detach()

        return torch.stack(all_goodness, dim=0).sum(dim=0), all_goodness[-1]
