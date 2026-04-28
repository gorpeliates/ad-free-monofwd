
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100 
from experiments.config import ExperimentConfig
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T
from typing import Tuple


def build_dataloaders(cfg: ExperimentConfig) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Builds train and test dataloaders based on the provided config.
    Args:
        cfg: ExperimentConfig object containing dataset and dataloader parameters.
    Returns:
        Tuples containing train, test loaders, number of input channels, and number of classes.

    """

    ds = cfg.dataset.lower()
    transform = T.ToTensor()
    
    if ds == "cifar10":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    elif ds == "cifar100":
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    
    # in_channels 1 for grayscale, 3 for RGB 
    if ds == "mnist":
        train_set = MNIST(cfg.data_root, train=True, download=True, transform=transform)
        test_set = MNIST(cfg.data_root, train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
    elif ds == "fashionmnist":
        train_set = FashionMNIST(cfg.data_root, train=True, download=True, transform=transform)
        test_set = FashionMNIST(cfg.data_root, train=False, download=True, transform=transform)
        in_channels, num_classes = 1, 10
    elif ds == "cifar10":
        train_set = CIFAR10(cfg.data_root, train=True, download=True, transform=transform)
        test_set = CIFAR10(cfg.data_root, train=False, download=True, transform=transform)
        in_channels, num_classes = 3, 10
    elif ds == "cifar100":
        train_set = CIFAR100(cfg.data_root, train=True, download=True, transform=transform)
        test_set = CIFAR100(cfg.data_root, train=False, download=True, transform=transform)
        in_channels, num_classes = 3, 100
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
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

    return train_loader, test_loader, in_channels, num_classes
