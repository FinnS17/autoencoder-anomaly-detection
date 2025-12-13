import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent
# Allow imports from project root when running the script directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoencoder import AutoEncoder
from conv_autoencoder import ConvAutoEncoder
from data import mnist_data_test
from helpers import get_device, make_noisy_images, set_seed

# --- Configuration: tweak here ---
MODEL_TYPE = "conv"  # "conv" or "mlp"
CHECKPOINT = None  # Path or None -> uses defaults below
ROWS = 4  # how many rows to show
NOISY_FRACTION = 0.1  # fraction of test set to corrupt
NOISE_LEVEL = 0.5  # stddev of Gaussian noise
SEED = 42

CHECKPOINT_NAMES = {
    "conv": ROOT / "conv_autoencoder.pth",
    "mlp": ROOT / "mlp_autoencoder.pth",
}
MODEL_FACTORY = {"conv": ConvAutoEncoder, "mlp": AutoEncoder}


def load_model(model_type: str, checkpoint: str, device: str):
    model_cls = MODEL_FACTORY[model_type]
    model = model_cls().to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    set_seed(SEED)
    device = get_device()

    ckpt_path = Path(CHECKPOINT) if CHECKPOINT else CHECKPOINT_NAMES[MODEL_TYPE]
    model = load_model(MODEL_TYPE, ckpt_path, device)

    noisy_indices, noisy_images = make_noisy_images(
        noisy_fraction=NOISY_FRACTION, noise_level=NOISE_LEVEL
    )
    rows = min(ROWS, len(noisy_indices))

    fig, ax = plt.subplots(rows, 4, figsize=(8, 2 * rows))

    for i in range(rows):
        clean_img, _ = mnist_data_test[noisy_indices[i]]
        noisy_img, _ = noisy_images[i]

        with torch.no_grad():
            clean_recon = model(clean_img.unsqueeze(0).to(device)).squeeze(0).cpu()
            noisy_recon = model(noisy_img.unsqueeze(0).to(device)).squeeze(0).cpu()

        ax[i, 0].imshow(clean_img.squeeze(), cmap="gray")
        ax[i, 0].set_title("Clean Input")
        ax[i, 0].axis("off")

        ax[i, 1].imshow(clean_recon.squeeze(), cmap="gray")
        ax[i, 1].set_title("Clean Recon")
        ax[i, 1].axis("off")

        ax[i, 2].imshow(noisy_img.squeeze(), cmap="gray")
        ax[i, 2].set_title("Noisy Input")
        ax[i, 2].axis("off")

        ax[i, 3].imshow(noisy_recon.squeeze(), cmap="gray")
        ax[i, 3].set_title("Noisy Recon")
        ax[i, 3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
