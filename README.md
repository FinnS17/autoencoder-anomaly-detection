# Autoencoder Playground (MNIST)

Kleines, übersichtliches Projekt, um Autoencoder auf dem MNIST-Datensatz zu trainieren, verrauschte Ziffern zu säubern und Rekonstruktionsfehler als Anomaliedetektor zu nutzen. Fokus: saubere Skripte, klare Namen, wenig Magie.

## Schnellstart
- Umgebung: `pip install -r requirements.txt`
- Alle Kommandos aus dem Repo-Root ausführen (Skripte hängen ihr eigenes `src` an `sys.path` an).
- Training starten: `python scripts/train.py` (Konfiguration direkt oben im Skript anpassen).
- Recon-Demo: `python scripts/demo_reconstruction.py` (Modell/Checkpoint/Rows oben anpassen).
- Rausch-Visualisierung: `python scripts/visualize_corrupted.py`.
- Rekonstruktionsfehler ansehen: `python scripts/reconstruction_error.py` (Threshold/Noise-Level oben anpassen).

## Struktur
- `scripts/` – Skripte für Training, Rekonstruktion, Rausch-Visualisierung, Anomaly-Check.
- Root: `autoencoder.py`, `conv_autoencoder.py`, `data.py`, `helpers.py`, `ssim_loss.py`.
- `data/` – MNIST-Download (wird automatisch erzeugt).
- `*.pth` – Checkpoints im Repo-Root (aus `.gitignore`).

## Hinweise
- Gerätewahl passiert automatisch (MPS > CUDA > CPU). Per Default wird ein Seed gesetzt (`--seed`), damit Rausch-Sampling und Initialisierung reproduzierbar sind.
- Modelle werden nach dem Training unter `conv_autoencoder.pth` bzw. `mlp_autoencoder.pth` gespeichert (override via `--checkpoint`).
- Das Skript lädt MNIST automatisch nach `./data`. Große Dateien/Checkpoints sind in `.gitignore`, damit das Repo schlank bleibt.

## Plots einfügen
- Loss-Kurven: MSE vs. SSIM (Konfiguration oben in `scripts/train.py` umstellen und erneut trainieren).
- Rekonstruktionen: Gegenüberstellung für `conv` und `mlp` (Clean/Noisy → Recon).
- Anomaly Detection: Histogramme der Rekonstruktionsfehler bei verschiedenen Noise-Levels/Thresholds.
