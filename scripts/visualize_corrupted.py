import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
# Allow imports from project root when running the script directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data import mnist_data_test
from helpers import make_noisy_images, set_seed

# --- Configuration: adjust to your liking ---
PAIRS = 4
NOISY_FRACTION = 0.1
NOISE_LEVEL = 0.3
SEED = 42


def main():
    set_seed(SEED)

    noisy_indices, noisy_images = make_noisy_images(
        noisy_fraction=NOISY_FRACTION, noise_level=NOISE_LEVEL
    )
    show_n = min(PAIRS, len(noisy_indices))

    fig, ax = plt.subplots(show_n, 2, figsize=(4, 2 * show_n))

    for i in range(show_n):
        original_img, _ = mnist_data_test[noisy_indices[i]]
        noisy_img, _ = noisy_images[i]

        ax[i, 0].imshow(original_img.squeeze(), cmap="gray")
        ax[i, 0].set_title("Original")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(noisy_img.squeeze(), cmap="gray")
        ax[i, 1].set_title("Corrupted")
        ax[i, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
