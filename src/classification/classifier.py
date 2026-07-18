"""Classify instruments in a Demucs "other" stem."""

import json
from pathlib import Path

import numpy as np
import torch

from src.classification.model import Model
from src.config import CHUNK_FRAMES, HOP_LENGTH, OPENMIC_DIR, SR
from src.feature_extraction.feature_extraction import extract_mel_spectrogram

# drop remainders shorter than ~3 seconds
MIN_REMAINDER_FRAMES = int(3 * SR / HOP_LENGTH)

# Demucs already covers these — don't report them from the CNN
DEAD_WEIGHT = {2, 5, 6, 19}  # bass, cymbals, drums, voice


def load_model(checkpoint_path, device):
    """Build Model, load weights, set eval mode."""
    model = Model().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def chunk_mel(mel):
    """Split (128, T) mel into a list of (1, 128, 431) tensors."""
    chunks = []
    t = mel.shape[1]
    start = 0

    while start < t:
        end = start + CHUNK_FRAMES
        piece = mel[:, start:end]
        width = piece.shape[1]

        if width == CHUNK_FRAMES:
            chunk = piece
        elif width >= MIN_REMAINDER_FRAMES:
            padded = np.zeros((mel.shape[0], CHUNK_FRAMES), dtype=mel.dtype)
            padded[:, :width] = piece
            chunk = padded
        else:
            break

        chunks.append(torch.from_numpy(chunk).unsqueeze(0).float())
        start = end

    return chunks


def predict_chunks(chunk_list, model, device):
    """Run all chunks in one batch; return probs shaped (C, 20)."""
    batch = torch.stack(chunk_list).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
    return probs.cpu()


def aggregate(probs):
    """Max-pool over chunks → song-level probs (20,)."""
    return probs.max(dim=0).values


def format_instruments(song_probs, threshold=0.5):
    """Threshold, drop dead-weight classes, attach names."""
    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    index_to_name = {index: name for name, index in class_map.items()}

    instruments = []
    for i, prob in enumerate(song_probs.tolist()):
        if i in DEAD_WEIGHT:
            continue
        if prob >= threshold:
            instruments.append({"name": index_to_name[i], "confidence": prob})
    return instruments


def classify(other_path, model, device):
    """other stem path → list of {name, confidence} for non-Demucs instruments."""
    mel = extract_mel_spectrogram(Path(other_path))
    chunks = chunk_mel(mel)
    if not chunks:
        return []
    probs = predict_chunks(chunks, model, device)
    song_probs = aggregate(probs)
    return format_instruments(song_probs)
