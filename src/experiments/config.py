from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "mlp"  # mlp | cnn
    pred_mode: str = "ff"  # ff | bp
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "./data"
    num_workers: int = 2
    activation: str = "relu" # relu | tanh 
    weight_decay: float = 1e-4  # Adam weight decay; equivalent to L2 regularization strength.
