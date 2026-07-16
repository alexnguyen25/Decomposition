"""Shared mel-spectrogram parameters for the classifier path."""

from pathlib import Path

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_FRAMES = 431

"""Shared parameters for training"""
NUM_EPOCHS = 25
OPENMIC_DIR = Path("data/openmic/openmic-2018")
CACHE_DIR = Path("data/openmic/mel_cache")
TRAIN_PARTITION = "train01.txt"
