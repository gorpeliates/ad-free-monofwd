from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "mlp"  # mlp | cnn
    batch_size: int = 256
    epochs: int = 200
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "./data"
    num_workers: int = 2
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    train_method: str = "all"  # autodiff | dd | backprop | all
    dd_eps: float = 1e-3
    dd_num_perturbations: int = 1
    tensorboard_logdir: str = "runs"
    optimizer: str = "adam"  # adam | sgd  (applies to AD/BP only, DD is only SGD with lr)

    def run_name(self, timestamp: str) -> str:
        base = f"{self.model}_{self.dataset}"

        if self.train_method == "dd":
            parts = f"dd_{self.optimizer}_{base}_lr{self.lr}_P{self.dd_num_perturbations}"
        elif self.train_method == "autodiff":
            parts = f"autodiff_{base}_{self.optimizer}"
        elif self.train_method == "backprop":
            parts = f"bp_{base}_{self.optimizer}"
        else:
            parts = f"all_{base}_{self.optimizer}"

        return f"{parts}_{timestamp}"
