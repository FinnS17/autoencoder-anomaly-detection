import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
# Allow imports from project root when running the script directly.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from autoencoder import AutoEncoder
from conv_autoencoder import ConvAutoEncoder
from data import get_dataloaders
from helpers import get_device, set_seed
from ssim_loss import SSIMLoss


MODEL_TYPE = "conv"  # "conv" or "mlp"
LOSS_TYPE = "ssim"  # "ssim" or "mse"
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 5e-2
MOMENTUM = 0.8
SEED = 42
CHECKPOINT = None  # Path or None -> uses default names below
SHOW_PLOT = True

CHECKPOINT_NAMES = {
    "conv": ROOT / "conv_autoencoder.pth",
    "mlp": ROOT / "mlp_autoencoder.pth",
}


def build_model(model_type: str) -> nn.Module:
    return ConvAutoEncoder() if model_type == "conv" else AutoEncoder()


def build_criterion(loss_name: str) -> nn.Module:
    return SSIMLoss() if loss_name == "ssim" else nn.MSELoss()


def train_epoch(model, loader, criterion, optimizer, device: str) -> float:
    model.train()
    epoch_loss = 0.0
    for x, _ in loader:
        x = x.to(device)
        loss = criterion(model(x), x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion, device: str) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            total_loss += criterion(model(x), x).item()
    return total_loss / len(loader)


def plot_history(train_losses: List[float], val_losses: List[float]):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs. Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    model = build_model(MODEL_TYPE).to(device)
    criterion = build_criterion(LOSS_TYPE)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    train_losses: List[float] = []
    val_losses: List[float] = []

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(
            f"Epoch {epoch + 1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

    ckpt_path = Path(CHECKPOINT) if CHECKPOINT else CHECKPOINT_NAMES[MODEL_TYPE]
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved weights to {ckpt_path}")

    if SHOW_PLOT:
        plot_history(train_losses, val_losses)


if __name__ == "__main__":
    main()
