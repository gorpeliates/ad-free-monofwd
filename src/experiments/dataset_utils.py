from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from experiments.config import ExperimentConfig
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from typing import Tuple
import torch


def build_dataloaders(
    cfg: ExperimentConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Builds train and test dataloaders based on the provided config.
    Args:
        cfg: ExperimentConfig object containing dataset and dataloader parameters.
    Returns:
        Tuples containing train, validation, test loaders, number of input channels, and number of classes.

    """

    ds = cfg.dataset.lower()
    transform = T.ToTensor()

    if ds == "cifar10":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                T.RandomHorizontalFlip(0.5),
                T.RandomCrop(32, padding=4),
            ]
        )
    elif ds == "cifar100":
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                T.RandomHorizontalFlip(0.5),
                T.RandomCrop(32, padding=4),
            ]
        )

    # in_channels 1 for grayscale, 3 for RGB
    if ds == "mnist":
        train_set = MNIST(cfg.data_root, train=True, download=False, transform=transform)
        test_set = MNIST(cfg.data_root, train=False, download=False, transform=transform)
        in_channels, num_classes = 1, 10
    elif ds == "fashionmnist":
        train_set = FashionMNIST(
            cfg.data_root, train=True, download=False, transform=transform
        )
        test_set = FashionMNIST(
            cfg.data_root, train=False, download=False, transform=transform
        )
        in_channels, num_classes = 1, 10
    elif ds == "cifar10":
        train_set = CIFAR10(
            cfg.data_root, train=True, download=False, transform=transform
        )
        test_set = CIFAR10(
            cfg.data_root, train=False, download=False, transform=transform
        )
        in_channels, num_classes = 3, 10
    elif ds == "cifar100":
        train_set = CIFAR100(
            cfg.data_root, train=True, download=False, transform=transform
        )
        test_set = CIFAR100(
            cfg.data_root, train=False, download=False, transform=transform
        )
        in_channels, num_classes = 3, 100
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    val_size = int(0.1 * len(train_set))
    train_size = len(train_set) - val_size
    train_set, val_set = random_split(
        train_set,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, in_channels, num_classes
