import torch
import torchvision.datasets as dsets
import torchvision.transforms as T

DATA_ROOT = "./data"

# they need to be downloaded first, run download_datasets.py if you haven't done so already
DATASETS = [
    ("MNIST", dsets.MNIST),
    ("FashionMNIST", dsets.FashionMNIST),
    ("CIFAR-10", dsets.CIFAR10),
    ("CIFAR-100", dsets.CIFAR100),
]

for name, cls in DATASETS:
    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")

    data = cls(DATA_ROOT, train=True, transform=T.ToTensor(), download=False)
    imgs = torch.stack([img for img, _ in data])  # (Num_samples, Channels, H, W)

    computed_mean = imgs.mean(dim=(0, 2, 3))
    computed_std = imgs.std(dim=(0, 2, 3))

    n_channels = computed_mean.shape[0]
    ch_labels = ["R", "G", "B"] if n_channels == 3 else ["gray"]

    print(f"  {'Channel':<8} {'Computed mean':>14} {'Computed std':>13}")
    print()
    for i, ch in enumerate(ch_labels):
        cm = computed_mean[i].item()
        cs = computed_std[i].item()
        print(f"  {ch:<8} {cm:>14.4f} {cs:>13.4f}")
