"""Shared configuration.

Everything tunable lives here and can be overridden by an environment variable,
so the same code runs on a laptop, in Colab, and on a host without edits.
"""

import os
from pathlib import Path

# --- mel-spectrogram parameters (the original CNN path) ---------------------
SR = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
CHUNK_FRAMES = 431

NUM_EPOCHS = 25
TRAIN_PARTITION = "split01_train.csv"
TEST_PARTITION = "split01_test.csv"

# --- dataset / checkpoint locations -----------------------------------------
if Path("/content").exists():
    _DRIVE = Path("/content/drive/MyDrive/Decomposition")
    OPENMIC_DIR = _DRIVE / "data" / "openmic" / "openmic-2018"
    CACHE_DIR = _DRIVE / "data" / "openmic" / "mel_cache"
    CHECKPOINT_PATH = _DRIVE / "models" / "classifier.pt"
else:
    OPENMIC_DIR = Path("data/openmic/openmic-2018")
    CACHE_DIR = Path("data/openmic/mel_cache")
    CHECKPOINT_PATH = Path("models/classifier.pt")

# --- classifier backend ------------------------------------------------------
# "beats" = frozen BEATs embeddings + trained MLP head (0.8045 tuned macro-F1
#           on OpenMIC-2018; the shipped default).
# "cnn"   = the original 2-conv scratch CNN (0.650). Kept because it is the
#           baseline the project improved on, and it needs no 345 MB download.
BACKEND = os.environ.get("BACKEND", "beats")

# Weights fetched by scripts/fetch_models.py live here.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "alexnguyen25/decomposition-models")

# "e10"          = the champion head (best on the OpenMIC test set).
# "stem_recalib" = variant fine-tuned on Demucs-processed clips. Measured to be
#                  statistically indistinguishable from e10 on real songs; kept
#                  as a documented negative result. See docs/research/.
CLASSIFIER_HEAD = os.environ.get("CLASSIFIER_HEAD", "e10")

HEAD_FILES = {
    "e10": "ckpt_E10_beats_head.pt",
    "stem_recalib": "ckpt_E10_stem_recalib.pt",
}

# --- BEATs inference protocol (must match how the head was trained) ----------
SR16 = 16000
WINDOW_SECONDS = 5           # BEATs windows
CHUNK_SECONDS = 10           # one classification unit = one OpenMIC clip length
WINDOW_SAMPLES = SR16 * WINDOW_SECONDS
CHUNK_SAMPLES = SR16 * CHUNK_SECONDS
EMBED_DIM = 768
NUM_CLASSES = 20

# --- song-level aggregation ---------------------------------------------------
# Mean of the top-K chunk probabilities, NOT max. Max-pooling turns a single
# noisy chunk into a confident song-level false positive — measured, see
# docs/research/2026-07-21-f1-improvement-research.md.
AGGREGATION_TOP_K = int(os.environ.get("AGGREGATION_TOP_K", "3"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))

# Demucs already returns these as their own stems, so the classifier must not
# also report them from the "other" stem.
DEAD_WEIGHT = {"bass", "cymbals", "drums", "voice"}

CLASS_MAP_PATH = Path(__file__).parent / "classification" / "assets" / "class-map.json"
