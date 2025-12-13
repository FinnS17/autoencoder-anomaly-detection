import torch
import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    """Lightweight convolutional autoencoder for MNIST images."""

    def __init__(self):
        super().__init__()

        # Downsample twice: 28x28 -> 14x14 -> 7x7
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # -> (16, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (16, 14, 14)
            nn.Conv2d(16, 32, 3, padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> (32, 7, 7)
        )

        # Upsample back to the original size.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                32, 16, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                16, 8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),  # -> (8, 28, 28)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=1),  # -> (1, 28, 28)
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        recon = self.decoder(encoded)
        return recon
