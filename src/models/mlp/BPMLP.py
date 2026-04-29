import torch
import torch.nn as nn

class BPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, activation: str = "relu"):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.LayerNorm(dims[i + 1]))
            activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
            layers.append(activation_fn)

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
