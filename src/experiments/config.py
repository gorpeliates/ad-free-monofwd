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
    dd_max_params_per_chunk: int = 50000  # max parameters perturbed at once; layers are divided into chunks of this size
    tensorboard_logdir: str = "runs"
    optimizer: str = "adam"  # adam | sgd  (applies to AD/BP only)
    cnn_proj_dim: int = 16  # FFZero channel-wise random projection dimension

    def run_name(self, timestamp: str) -> str:
        base = f"{self.model}_{self.dataset}"

        if self.train_method == "dd":
            parts = f"freeze_dd_{base}_maxp{self.dd_max_params_per_chunk}_P{self.dd_num_perturbations}"
        elif self.train_method == "autodiff":
            parts = f"autodiff_{base}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"
        elif self.train_method == "backprop":
            parts = f"bp_{base}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"
        else:
            parts = f"all_{base}_{self.optimizer}_lr{self.lr}_bs{self.batch_size}"

        return f"{parts}_{timestamp}"
