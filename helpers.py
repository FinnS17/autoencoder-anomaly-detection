import random
from typing import List, Sequence, Tuple

import numpy as np
import torch

from data import mnist_data_test


def get_device() -> str:
    """Pick the best available device."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def add_noise(x: torch.Tensor, noise_level: float = 0.075) -> torch.Tensor:
    noise = torch.randn_like(x) * noise_level
    x_noisy = x + noise
    # Keep pixel values in [0, 1] after adding noise.
    x_noisy = torch.clamp(x_noisy, 0.0, 1.0)
    return x_noisy


def make_noisy_images(
    noisy_fraction: float = 0.1, noise_level: float = 0.075
) -> Tuple[List[int], List[Tuple[torch.Tensor, int]]]:
    """
    Sample a subset of test images, corrupt them with Gaussian noise, and return
    both indices and noisy copies.
    """
    num_test = len(mnist_data_test)
    num_noisy = max(1, min(num_test, int(noisy_fraction * num_test)))
    noisy_indices: Sequence[int] = random.sample(range(num_test), num_noisy)

    noisy_images: List[Tuple[torch.Tensor, int]] = []
    for idx in noisy_indices:
        img, label = mnist_data_test[idx]
        corrupted = add_noise(img, noise_level=noise_level)
        noisy_images.append((corrupted, label))

    return list(noisy_indices), noisy_images
