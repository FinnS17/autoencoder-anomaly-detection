import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

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
THRESHOLD = 0.003  # decision boundary on reconstruction error
NOISY_FRACTION = 0.1  # fraction of test set to corrupt
NOISE_LEVEL = 0.075  # stddev for Gaussian noise
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


def reconstruction_errors(
    model, images: List[torch.Tensor], device: str
) -> List[float]:
    criterion = torch.nn.MSELoss(reduction="none")
    errors: List[float] = []
    for x in images:
        x = x.unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(x)
        error = criterion(recon, x).mean().item()
        errors.append(error)
    return errors


def main():
    set_seed(SEED)
    device = get_device()

    ckpt_path = Path(CHECKPOINT) if CHECKPOINT else CHECKPOINT_NAMES[MODEL_TYPE]
    model = load_model(MODEL_TYPE, ckpt_path, device)

    noisy_indices, noisy_images = make_noisy_images(
        noisy_fraction=NOISY_FRACTION, noise_level=NOISE_LEVEL
    )
    all_indices = list(range(len(mnist_data_test)))
    clean_indices = [idx for idx in all_indices if idx not in noisy_indices]

    clean_images = [mnist_data_test[idx][0] for idx in clean_indices]
    corrupted_images = [pair[0] for pair in noisy_images]

    errors_clean = reconstruction_errors(model, clean_images, device)
    errors_corrupted = reconstruction_errors(model, corrupted_images, device)

    y_true = np.array([0] * len(errors_clean) + [1] * len(errors_corrupted))
    y_pred = np.array(
        [1 if e > THRESHOLD else 0 for e in errors_clean + errors_corrupted]
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    plt.figure(figsize=(6, 4))
    plt.hist(errors_clean, bins=50, alpha=0.6, label="Clean Recon", density=True)
    plt.hist(errors_corrupted, bins=50, alpha=0.6, label="Corrupted Recon", density=True)
    plt.axvline(THRESHOLD, color="red", linestyle="--", label="Threshold")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Density")
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
