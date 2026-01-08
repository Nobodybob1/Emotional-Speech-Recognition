# Emotional Speech Recognition

A small PyTorch notebook project for emotion classification from speech using **log-mel spectrograms**.

The notebook contains workflows for:
- **RAVDESS** (expected under `archive/Actor_XX/`)
- **CREMA-D** (expected under `AudioWAV/`)

## What’s in this repo

- `main.ipynb` – end-to-end workflow:
  - load datasets from `archive/` (RAVDESS) and/or `AudioWAV/` (CREMA-D)
  - parse emotion labels from filenames
  - compute fixed-size log-mel spectrograms
  - train/val/test split by **actor IDs**
  - train a small 2D CNN classifier in PyTorch
  - evaluate with confusion matrix + classification report
- `archive/` – RAVDESS folder expected to contain `Actor_01` … `Actor_24` subfolders with `.wav` files
- `AudioWAV/` – CREMA-D audio folder (flat directory of `.wav` files)
- `logmel_data/` – cached `.npy` log-mel tensors (train/val/test)

## Requirements

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then open and run the notebook:

```bash
jupyter notebook main.ipynb
```

(If you don’t have Jupyter installed, run `pip install notebook` or use VS Code’s notebook support.)

## Dataset notes

### RAVDESS

This notebook expects the RAVDESS audio-only speech files named like:

`03-01-06-01-02-01-12.wav`

Where the **3rd field** is the emotion code:

- `01` neutral
- `02` calm
- `03` happy
- `04` sad
- `05` angry
- `06` fearful
- `07` disgust
- `08` surprised

Place the dataset so your folder structure looks like:

```
Emotional Speech/
  main.ipynb
  archive/
    Actor_01/
      *.wav
    Actor_02/
      *.wav
    ...

### CREMA-D

The notebook also includes a loader for CREMA-D where audio files are expected under:

```
Emotional Speech/
  AudioWAV/
    *.wav
```
```

## Model & preprocessing (as implemented)

- Audio is loaded as mono and padded/trimmed to a fixed duration (see notebook cells)
- Log-mel spectrogram parameters:
  - `n_mels = 64`, `n_fft = 1024`, `hop_length = 160`
  - time dimension is padded/trimmed to a fixed number of frames
  - per-sample normalization
- Model: a small CNN classifier in PyTorch (plus some experiment cells using `torchvision` backbones)

## Reproducibility

- The train/test split is done by shuffling actor IDs with a fixed RNG seed.

## License / attribution

- The dataset is not included in this repository.
- RAVDESS is distributed under its own license terms—please consult the official RAVDESS license and cite the authors when using the dataset.
