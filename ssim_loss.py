import torch
import torch.nn as nn


class SSIMLoss(nn.Module):
    """Structural Similarity as a loss (1 - SSIM)."""

    def __init__(self, L=1.0, eps=1e-12):
        super().__init__()
        self.C1 = (0.01 * L) ** 2  # stability terms to avoid divide by zero
        self.C2 = (0.03 * L) ** 2
        self.eps = eps

    def forward(self, x, y):
        # x, y: (B, 1, 28, 28)
        B = x.shape[0]

        # flatten per image
        x = x.view(B, -1)
        y = y.view(B, -1)

        # means
        mu_x = x.mean(dim=1, keepdim=True)
        mu_y = y.mean(dim=1, keepdim=True)

        # variances
        x_mu = x - mu_x
        y_mu = y - mu_y

        var_x = (x_mu ** 2).mean(dim=1, keepdim=True)
        var_y = (y_mu ** 2).mean(dim=1, keepdim=True)

        # covariance
        cov_xy = (x_mu * y_mu).mean(dim=1, keepdim=True)

        # constants
        C1, C2, eps = self.C1, self.C2, self.eps

        # SSIM numerator / denominator
        num = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)
        den = (mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2)

        ssim_per_image = num / (den + eps)
        ssim = ssim_per_image.mean()

        return 1 - ssim
