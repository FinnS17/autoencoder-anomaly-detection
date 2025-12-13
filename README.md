# Autoencoder Playground (MNIST)

Project to understand how autoencoders work, compare an MLP vs. a small Conv autoencoder on MNIST, denoise noisy digits, and use reconstruction error as a quick anomaly detector. 
## What I looked at
- MLP vs. Conv autoencoder: which reconstructs MNIST digits better with few parameters?
- Denoising: how much Gaussian noise can each model clean up?
- Anomaly detection: can a simple reconstruction-error threshold flag corrupted digits?

## Quickstart
- Install: `pip install -r requirements.txt`
- Run from repo root.
- Train: `python scripts/train.py` (edit config block: model, loss, epochs, lr, seed).
- Recon demo: `python scripts/demo_reconstruction.py` (rows/noise/model at top).
- Noise visualization: `python scripts/visualize_corrupted.py`.
- Reconstruction error / anomaly check: `python scripts/reconstruction_error.py` (threshold/noise at top).

## Training (`scripts/train.py`)
- Config: `MODEL_TYPE` (`conv`/`mlp`), `LOSS_TYPE` (`ssim`/`mse`), `EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`, `MOMENTUM`, `SEED`, `CHECKPOINT`, `SHOW_PLOT`.
- Output: saves `conv_autoencoder.pth` or `mlp_autoencoder.pth`; prints train/val loss per epoch; optional loss plot.

## Demos and plots
- **Original vs. corrupted** (`scripts/visualize_corrupted.py`): shows how much Gaussian noise was added.
  <img src="bilder_autoencoder/corrupted.png" width="55%">
  *Read it:* Left = clean digits, right = corrupted counterparts.

- **Reconstructions** (`scripts/demo_reconstruction.py`): clean/noisy inputs and model outputs side by side.
  <img src="bilder_autoencoder/reconstruction_clean_corrupted.png" width="65%">
  *Read it:* Columns = Clean Input | Clean Recon | Noisy Input | Noisy Recon. Conv generally preserves strokes better than the MLP when noise is present.

## Anomaly detection (`scripts/reconstruction_error.py`)
- Corrupt part of the test set, compute reconstruction MSE, print a confusion matrix, and plot error histograms.
  <p align="center">
    <img src="bilder_autoencoder/recon_error_n0.075_t.png" width="45%">
    <img src="bilder_autoencoder/recon_error_n0.4.png" width="45%">
  </p>
  *Read it:* Clean samples cluster at low error; noisy samples shift the distribution right. Higher noise widens the gap. Choose `THRESHOLD` where the curves separate best.

## Findings
- The Conv autoencoder reconstructs MNIST more cleanly and is more robust to noise than the MLP.
- SSIM loss tends to produce smoother, cleaner outputs than plain MSE on noisy inputs.
- A simple reconstruction-error threshold works as a toy anomaly detector: low false positives at moderate noise; at heavy noise you need a higher threshold.

## Project structure
- `scripts/` – train, reconstruction demo, noise visualization, reconstruction-error thresholding.
- Root – `autoencoder.py`, `conv_autoencoder.py`, `data.py`, `helpers.py`, `ssim_loss.py`.
- `data/` – MNIST download/cache (auto-created).
- `*.pth` – checkpoints in repo root (ignored by git).

## Notes
- Device selection is automatic: MPS > CUDA > CPU. Reproducibility via `SEED` in each script.
- MNIST downloads to `./data`. Large artifacts (data/checkpoints) are ignored via `.gitignore`.
