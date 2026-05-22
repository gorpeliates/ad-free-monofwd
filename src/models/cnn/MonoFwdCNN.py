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
        proj_dim: int = 16,
    ):
        super().__init__()
        # FFzero CNN: kernel 6x6, stride 1, padding 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=6, stride=1, padding=2)
        self.bn = nn.BatchNorm2d(out_ch)

        self.out_ch = out_ch
        self.proj_dim = proj_dim
        self.num_classes = num_classes

        # Trainable prototype matrix: [out_ch * proj_dim] -> num_classes
        self.M = nn.Parameter(torch.empty(out_ch * proj_dim, num_classes))
        nn.init.kaiming_uniform_(self.M, a=math.sqrt(5))

        # Fixed random projection matrices, one per channel: [C, proj_dim, H*W]
        # Registered as a buffer, not trainabke
        #  initialized lazily on first forward pass
        self.register_buffer('A', None)

    def _init_projection(self, spatial_size: int, device: torch.device) -> None:
        # A[c]: [proj_dim, H*W] random projection for channel c
        A = torch.randn(self.out_ch, self.proj_dim, spatial_size, device=device)
        # scale for normalization
        A = A / math.sqrt(spatial_size)
        self.register_buffer('A', A)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Conv -> BN -> ReLU -> MaxPool
        Returns (next_h, goodness).
        """
        x = self.conv(x)
        x = self.bn(x)
        a = F.relu(x)
        next_h = F.max_pool2d(a, kernel_size=2, stride=2)

        B, C, H, W = a.shape
        spatial_size = H * W

        # lazy init check
        if self.A is None or self.A.shape[-1] != spatial_size:
            self._init_projection(spatial_size, a.device)

        # u: [B, C, H*W] — flatten spatial dims per channel
        u = a.view(B, C, spatial_size)

        # z: [B, C, proj_dim] — channel-wise random projection
        
        z = torch.einsum('bcs,cps->bcp', u, self.A)

        # flatten all channel projections -> [B, C*proj_dim]
        z_flat = z.reshape(B, C * self.proj_dim)

        # g: [B, num_classes]
        g = z_flat @ self.M

        return next_h, g


class MonoFwdCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        channels: List[int],
        num_classes: int,
        proj_dim: int = 16,
    ):
        super().__init__()
        self.num_classes = num_classes
        dims = [in_ch] + channels
        self.blocks = nn.ModuleList(
            [
                MonoFwdConvBlock(
                    dims[i], dims[i + 1], num_classes, proj_dim=proj_dim
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
