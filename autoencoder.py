import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    """Simple MLP autoencoder for MNIST (works on flattened 28x28 images)."""

    def __init__(self):
        super().__init__()

        # Encoder shrinks 784 -> 32 as a compact code.
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Decoder expands the code back to 784 pixels.
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),  # keep pixels in [0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        return recon.view(-1, 1, 28, 28)  # reshape back to image tensor
