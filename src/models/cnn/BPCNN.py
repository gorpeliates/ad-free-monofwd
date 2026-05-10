from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return F.max_pool2d(x, kernel_size=2)


class BPCNN(nn.Module):
    def __init__(self, in_ch: int, channels: List[int], num_classes: int):
        super().__init__()
        dims = [in_ch] + channels
        self.blocks = nn.Sequential(
            *[BPConvBlock(dims[i], dims[i + 1]) for i in range(len(channels))]
        )
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.classifier(x)
