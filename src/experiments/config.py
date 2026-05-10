from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "mlp"  # mlp | cnn
    pred_mode: str = "ff"  # ff | bp
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "./data"
    num_workers: int = 2
    activation: str = "relu"  # relu | tanh
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-4
    training_method: str = "autodiff"  # autodiff | dd
    dd_eps: float = 1e-3
    dd_num_perturbations: int = 1
    tensorboard_logdir: str = "runs"
    scheduler_step_size: int = 15
