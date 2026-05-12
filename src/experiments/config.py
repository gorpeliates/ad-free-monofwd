from dataclasses import dataclass
import torch


@dataclass
class ExperimentConfig:
    dataset: str = "mnist"
    model: str = "mlp"  # mlp | cnn
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    data_root: str = "./data"
    num_workers: int = 2
    early_stopping: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1e-4
    training_method: str = "all"  # autodiff | dd | backprop | all
    dd_eps: float = 1e-3
    dd_num_perturbations: int = 2
    tensorboard_logdir: str = "runs"
    scheduler_step_size: int = 30
    optimizer: str = "adam"  # adam | sgd

    def run_name(self, timestamp: str) -> str:
        base = f"{self.model}_{self.dataset}"
        if self.model == "mlp":
            base += f"_step{self.scheduler_step_size}"

        if self.training_method == "dd":
            parts = f"dd_{base}_eps{self.dd_eps}_P{self.dd_num_perturbations}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"
        elif self.training_method == "autodiff":
            parts = f"autodiff_{base}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"
        elif self.training_method == "backprop":
            parts = f"bp_{base}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"
        else:
            parts = f"all_{base}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"

        return f"{parts}_{timestamp}"
