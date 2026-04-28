# MONO-FORWARD MLP

import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Tuple
from experiments.dataset_utils import build_dataloaders

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
    
    def local_losses_logits(self, x:torch.Tensor, y:torch.Tensor) ->  Tuple[List[torch.Tensor], List[torch.Tensor]]:
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
    

def train_monofwd_mlp_one_epoch_autodiff(
        model: MonoFwdMLP, 
        optimizers: List[torch.optim.Optimizer], 
        dataloader: torch.utils.data.DataLoader, 
        device:str = "cuda",
        pred_mode: str = "ff"
    ) -> Tuple[float, float]:
    """
    Train the MonoFwdMLP for one epoch using automatic differentiation for updates in the projection matrices.
    Returns:
      Average loss, accuracy 
    """
    model.to(device)
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for x,y in dataloader:
        x : torch.Tensor
        y : torch.Tensor

        x = x.to(device,non_blocking=True)
        y = y.to(device,non_blocking=True)


        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        
        losses, logits_per_layer = model.local_losses_logits(x, y)

        # updates model weights.
        for loss in losses:
            loss.backward()

        for opt in optimizers:
            opt.step()

        # no grad for evaluation
        with torch.no_grad():
            if pred_mode == "ff":
                final_goodness = torch.stack(logits_per_layer, dim=0).sum(dim=0)
            elif pred_mode == "bp":
                final_goodness = logits_per_layer[-1]
            else:
                raise ValueError(f"Unknown pred_mode: {pred_mode}")

            total_loss += sum(loss.item() for loss in losses) * x.size(0)
            total_correct += int((final_goodness.argmax(dim=1) == y).sum().item())
            total_seen += x.size(0)

    return total_loss / total_seen, total_correct / total_seen

