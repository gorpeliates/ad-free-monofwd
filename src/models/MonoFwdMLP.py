# MONO-FORWARD MLP

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple


class MonoFwdLinearBlock(nn.Module):
    def __init__(self,in_dim: int, out_dim : int, num_classes: int, activation:str = "relu"):
        super().__init__()    
        self.linear = nn.Linear(in_dim, out_dim)

        # the projection matrix, where m = num categories, n = number of neurons
        m = num_classes
        n = out_dim
        self.M = nn.Parameter(torch.randn(m,n))

        self.activation =  F.relu if activation == "relu" else F.tanh
        

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        # a -> activation, g -> goodness
        # goodness is just the score of each category
        z = self.linear(x)
        a = self.activation(z) 
        g = a @ self.M.T
        return a,g

class MonoFwdMLP(nn.Module):

    def __init__(self,input_dim : int, hidden_dims: list[int], num_classes: int, activation : str = "relu"):
        super().__init__()    
        dims = [input_dim] + hidden_dims
        self.blocks = nn.ModuleList(
            [MonoFwdLinearBlock(dims[i], dims[i + 1], num_classes, activation=activation) for i in range(len(hidden_dims))]
        )
        self.num_classes = num_classes
    
    def local_losses_logits(self, x:torch.Tensor, y:torch.Tensor):
        """
        Returns the local losses for each block.
        The loss for each block is the cross-entropy loss between the goodness scores and the true labels,
        as mentioned in the paper. 

        The activations are detached to prevent gradients from flowing back through the previous blocks,
        which allows each block to be trained independently.

        
        """
        
        if x.dim() > 2:
            x = x.flatten(1)
        
        losses: List[torch.Tensor] = []
        logits: List[torch.Tensor] = []
        
        h = x
        for block in self.blocks:
            block: MonoFwdLinearBlock
            a,g = block.forward(h)
            losses.append(F.cross_entropy(g,y)) # this already does softmax 
            logits.append(g)
            # we detach the activations to prevent gradients from flowing back through the previous blocks
            # this is the key for training each block independently
            h = a.detach()
        
        return losses,logits
    
    
    @torch.no_grad()
    def predict_logits(self, x: torch.Tensor, mode: str = "ff") -> torch.Tensor:
        """
            Predicts the logits for the input x using either feedforward (ff) or backpropagation (bp) mode.
            FF mode -> sums the goodness scores from all blocks to make a prediction.
            BP mode -> uses the goodness scores from the last block to make a prediction.
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
        if mode == "ff":
            return torch.stack(all_logits, dim=0).sum(dim=0)
        if mode == "bp":
            return all_logits[-1]
        raise ValueError(f"Unknown prediction mode: {mode}")
    

def train_MonoFwdMLP_AD_One_Epoch(
        model: MonoFwdMLP, 
        optimizers: List[torch.optim.Optimizer], 
        dataloader: torch.utils.data.DataLoader, 
        device:str = "cuda"
    ) -> Tuple[float, float]:
    """
    Train the MonoFwdMLP for one epoch using automatic differentiation for updates in the projection matrices.
    Returns:
      Average loss, accuracy 
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_seen = 0

    for x,y in dataloader:
        x : torch.Tensor
        y : torch.Tensor

        x.to(device,non_blocking=True)
        y.to(device,non_blocking=True)

        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        losses, logits_per_layer = model.local_losses_logits(x, y)

        # updates model weights.
        for loss in losses:
            loss.backward()

        for opt in optimizers:
            opt.step()
    
    final_logits = torch.stack(logits_per_layer, dim=0).sum(dim=0)
    total_loss += sum(float(loss.item()) for loss in losses) * x.size(0)
    total_correct += int((final_logits.argmax(dim=1) == y).sum().item())
    total_seen = 0

    return total_loss / total_seen, total_correct / total_seen


def run_experiment(cfg: ExperimentConfig) -> None:
    set_seed(cfg.seed)
    train_loader, test_loader, in_channels, num_classes = build_dataloaders(cfg)
    model = build_model(cfg, in_channels, num_classes).to(cfg.device)

    # Lazy conv classifiers are created on first forward pass.
    # Run one batch through local_losses first so all parameters exist.
    xb, yb = next(iter(train_loader))
    xb = xb.to(cfg.device)
    yb = yb.to(cfg.device)
    _ = model.local_losses(xb, yb)

    optimizers = build_optimizers(model, cfg)

    best_acc = -math.inf
    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, optimizers, train_loader, cfg.device)
        test_loss, test_acc = evaluate(model, test_loader, cfg.device, cfg.pred_mode)
        best_acc = max(best_acc, test_acc)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
            f"best_test_acc={best_acc:.4f}"
        )