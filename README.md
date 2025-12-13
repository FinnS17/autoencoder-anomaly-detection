# Autoencoder Playground (MNIST)

Small PyTorch playground to compare a simple MLP autoencoder vs. a small convolutional autoencoder on MNIST: reconstruction quality, denoising, and toy anomaly detection via reconstruction error.

## What’s inside
- MLP vs. Conv autoencoder (reconstruction quality + loss curves)
- Denoising (Gaussian noise on MNIST digits)
- Anomaly detection (threshold on reconstruction MSE + confusion matrix)

## Setup
    pip install -r requirements.txt

Run scripts from the repo root.

## Scripts
### Train
    python scripts/train.py

Edit the config block at the top (model type, loss, epochs, lr, seed, etc.).
Saves a checkpoint (conv_autoencoder.pth / mlp_autoencoder.pth, ignored by git).

### Visualize corruption level
    python scripts/visualize_corrupted.py

### Reconstructions demo (clean + noisy)
    python scripts/demo_reconstruction.py

### Reconstruction error + anomaly detection
    python scripts/reconstruction_error.py

Computes reconstruction MSE for clean vs. corrupted test samples, prints TP/TN/FP/FN, plots the confusion matrix, and shows error histograms.
Tune THRESHOLD, NOISY_FRACTION, NOISE_LEVEL, MODEL_TYPE, SEED in the config block.

## Results (plots)
### Original vs. corrupted
<img src="bilder_autoencoder/corrupted.png" width="55%">

### Reconstructions
<img src="bilder_autoencoder/reconstruction_clean_corrupted.png" width="70%">

### Reconstruction error distributions (thresholding)
<table>
  <tr>
    <td align="center"><b>overlap (noise = 0.075)</b></td>
    <td align="center"><b>almost no overlap (noise = 0.4)</b></td>
  </tr>
  <tr>
    <td align="center"><img src="bilder_autoencoder/recon_error_n0.075_t.png" height="220"></td>
    <td align="center"><img src="bilder_autoencoder/recon_error_n0.4.png" height="220"></td>
  </tr>
</table>

## Notes / takeaways
- The Conv autoencoder reconstructs MNIST digits cleaner and is more robust to noise than the MLP.
- SSIM loss tends to focus more on structure (strokes), while MSE optimizes pixel-wise error.
- Thresholding reconstruction error works as a simple anomaly detector:
  moderate noise -> overlap (meaningful trade-off), heavy noise -> trivial separation.

## Project structure
- scripts/ – training, demos, corruption visualization, thresholding + metrics
- autoencoder.py, conv_autoencoder.py – model definitions
- ssim_loss.py – SSIM loss implementation (no built-in SSIM used)
- data.py, helpers.py – MNIST loading + noise helpers
- bilder_autoencoder/ – figures for README
- data/, *.pth – generated locally (ignored)

## Device
Device selection is automatic: MPS > CUDA > CPU.
Reproducibility via SEED in each script.