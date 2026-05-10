#!/usr/bin/env python3

from torchvision.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
import torchvision.transforms as T

data_root = "./data"
transform = T.ToTensor()

print("Downloading MNIST...")
MNIST(data_root, train=True, download=True, transform=transform)
MNIST(data_root, train=False, download=True, transform=transform)

print("Downloading FashionMNIST...")
FashionMNIST(data_root, train=True, download=True, transform=transform)
FashionMNIST(data_root, train=False, download=True, transform=transform)

print("Downloading CIFAR10...")
CIFAR10(data_root, train=True, download=True, transform=transform)
CIFAR10(data_root, train=False, download=True, transform=transform)

print("Downloading CIFAR100...")
CIFAR100(data_root, train=True, download=True, transform=transform)
CIFAR100(data_root, train=False, download=True, transform=transform)

print("Done!")
