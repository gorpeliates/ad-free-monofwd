import random

import numpy as np
import torch

from src.experiments import ExperimentConfig

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print("Running experiment with config:")
    cfg = ExperimentConfig()


if __name__ == "__main__":
    main()
