import torch
import torch.nn as nn

class BPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, dropout: float = 0.0):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
