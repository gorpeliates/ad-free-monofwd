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
        use_bn: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else nn.Identity()

        # proj matrix for goodness scores, use global avg pool
        # TODO consider other flattening methods
        m = num_classes
        n = out_ch
        self.M = nn.Parameter(torch.empty(m, n))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))

        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, conv layer, optional BN, ReLU and average pooling
        """
        x = self.conv(x)
        x = self.bn(x)
        a = F.relu(x)
        x = F.avg_pool2d(a, kernel_size=2)  # Average pooling for the next layer

        # Global average pool for the goodness score calculation
        # dimensions are (batch_size, out_ch, H, W)
        # when we global avg pool, we essentially map it to (batch_size, out_ch,1,1)
        # when we flatten at axis 1 (which is the out_ch) we get (batch_size, out_ch)
        pooled_a = F.adaptive_avg_pool2d(a, (1, 1)).flatten(1)
        g = pooled_a @ self.M.T

        return x, g


class MonoFwdCNN(nn.Module):
    def __init__(
        self, in_ch: int, channels: List[int], num_classes: int, use_bn: bool = False
    ):
        super().__init__()
        self.num_classes = num_classes
        dims = [in_ch] + channels
        self.blocks = nn.ModuleList(
            [
                MonoFwdConvBlock(dims[i], dims[i + 1], num_classes, use_bn=use_bn)
                for i in range(
                    len(channels),
                )
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
        Predicts logits using the same convention as MonoFwdMLP.
        FF mode sums the goodness scores from all blocks.
        BP mode uses the goodness scores from the last block.
        """
        h = x
        all_logits = []

        for block in self.blocks:
            block: MonoFwdConvBlock
            a, g = block.forward(h)
            all_logits.append(g)
            h = a

        return torch.stack(all_logits, dim=0).sum(dim=0), all_logits[-1]
