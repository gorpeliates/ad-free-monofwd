import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class BPMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], num_classes: int, activation: str = "relu"):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            activation_fn = nn.ReLU() if activation == "relu" else nn.Tanh()
            layers.append(activation_fn)

        layers.append(nn.Linear(hidden_dims[-1], num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.flatten(1)
        return self.net(x)
    
def train_bp_mlp_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x, y in dataloader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        goodness = model(x)
        loss = F.cross_entropy(goodness, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += int((goodness.argmax(dim=1) == y).sum().item())
        total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen