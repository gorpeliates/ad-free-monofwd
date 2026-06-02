import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from models.mlp.BPMLP import BPMLP
from models.mlp.MonoFwdMLP import MonoFwdMLP
from models.cnn.BPCNN import BPCNN
from models.cnn.MonoFwdCNN import MonoFwdCNN

DATASETS = {
    "mnist": {
        "mlp_in": 28 * 28,
        "mlp_hidden": [50] * 2,
        "cnn_in_ch": 1,
        "cnn_channels": [32, 32],
        "num_classes": 10,
    },
    "fashionmnist": {
        "mlp_in": 28 * 28,
        "mlp_hidden": [100] * 3,
        "cnn_in_ch": 1,
        "cnn_channels": [64, 64],
        "num_classes": 10,
    },
    "cifar10": {
        "mlp_in": 32 * 32 * 3,
        "mlp_hidden": [200] * 4,
        "cnn_in_ch": 3,
        "cnn_channels": [128, 128],
        "num_classes": 10,
    },
    "cifar100": {
        "mlp_in": 32 * 32 * 3,
        "mlp_hidden": [200] * 4,
        "cnn_in_ch": 3,
        "cnn_channels": [128, 128],
        "num_classes": 100,
    },
}

CNN_PROJ_DIM = 2048


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    header = f"{'Dataset':<14} {'Model':<12} {'Params':>12}"
    print(header)
    print("-" * len(header))

    for ds, cfg in DATASETS.items():
        models = {
            "BPMLP": BPMLP(cfg["mlp_in"], cfg["mlp_hidden"], cfg["num_classes"]),
            "MonoFwdMLP": MonoFwdMLP(cfg["mlp_in"], cfg["mlp_hidden"], cfg["num_classes"]),
            "BPCNN": BPCNN(cfg["cnn_in_ch"], cfg["cnn_channels"], cfg["num_classes"]),
            "MonoFwdCNN": MonoFwdCNN(
                cfg["cnn_in_ch"],
                cfg["cnn_channels"],
                cfg["num_classes"],
                proj_dim=CNN_PROJ_DIM,
            ),
        }
        for name, model in models.items():
            print(f"{ds:<14} {name:<12} {count_params(model):>12,}")
        print()


if __name__ == "__main__":
    main()
