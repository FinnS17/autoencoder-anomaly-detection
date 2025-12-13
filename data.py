from pathlib import Path

import torch
from torchvision import datasets, transforms

transform = transforms.ToTensor()
# Keep downloads local to the project folder.
DATA_ROOT = Path(__file__).resolve().parent / "data"

mnist_data_train = datasets.MNIST(
    root=DATA_ROOT, train=True, download=True, transform=transform
)
mnist_data_test = datasets.MNIST(
    root=DATA_ROOT, train=False, download=True, transform=transform
)


def get_dataloaders(batch_size: int = 64):
    """Return train/test loaders with the shared MNIST transform."""
    train_loader = torch.utils.data.DataLoader(
        dataset=mnist_data_train, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=mnist_data_test, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


train_loader, test_loader = get_dataloaders()
