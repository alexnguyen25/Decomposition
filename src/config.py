"""Shared mel-spectrogram parameters for the classifier path."""

from pathlib import Path

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_FRAMES = 431

NUM_EPOCHS = 25
TRAIN_PARTITION = "split01_train.csv"

if Path("/content").exists():
    _DRIVE = Path("/content/drive/MyDrive/Decomposition")
    OPENMIC_DIR = _DRIVE / "data" / "openmic" / "openmic-2018"
    CACHE_DIR = _DRIVE / "data" / "openmic" / "mel_cache"
    CHECKPOINT_PATH = _DRIVE / "models" / "classifier.pt"
else:
    OPENMIC_DIR = Path("data/openmic/openmic-2018")
    CACHE_DIR = Path("data/openmic/mel_cache")
    CHECKPOINT_PATH = Path("models/classifier.pt")
