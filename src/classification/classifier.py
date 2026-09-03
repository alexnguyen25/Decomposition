"""Identify instruments in a Demucs "other" stem.

Two backends behind one interface:

  "beats" (default) — frozen BEATs embeddings + the trained MLP head, 0.8045
                      tuned macro-F1 on OpenMIC-2018.
  "cnn"             — the original 2-conv scratch CNN, 0.650. Kept because it is
                      the baseline this project improved on, and because it
                      needs no 345 MB download.

Both share the same song-level aggregation and formatting, which matters: the
over-prediction problem was never purely a model problem (see ``aggregate``).
"""

import json
from pathlib import Path

import librosa
import numpy as np
import torch

from src.classification.embedder import (
    embed_chunks,
    load_backbone,
    pick_device,
    to_chunks,
)
from src.classification.head import load_head
from src.classification.model import Model
from src.config import (
    AGGREGATION_TOP_K,
    BACKEND,
    CHECKPOINT_PATH,
    CHUNK_FRAMES,
    CLASS_MAP_PATH,
    CLASSIFIER_HEAD,
    CONFIDENCE_THRESHOLD,
    DEAD_WEIGHT,
    HEAD_FILES,
    HOP_LENGTH,
    MODEL_DIR,
    SR,
    SR16,
)
from src.feature_extraction.feature_extraction import extract_mel_spectrogram

MIN_REMAINDER_FRAMES = int(3 * SR / HOP_LENGTH)

with open(CLASS_MAP_PATH) as _f:
    CLASS_MAP: dict[str, int] = json.load(_f)
INDEX_TO_NAME = {index: name for name, index in CLASS_MAP.items()}


# --- aggregation & formatting (shared by both backends) ----------------------

def aggregate(chunk_probs: np.ndarray, top_k: int = AGGREGATION_TOP_K) -> np.ndarray:
    """Chunk probabilities (n_chunks, 20) -> one song-level vector (20,).

    Mean of the top-K chunks per class, NOT max. Max-pooling asks "was this
    instrument ever confidently heard in any 10 s window?", which one noisy
    chunk answers yes to — measured on real songs, the CNN with max-pooling
    returned 13 instruments for a 3-instrument track. Requiring K chunks to
    agree makes a single bad window unable to carry a class on its own.
    """
    k = min(top_k, len(chunk_probs))
    return np.sort(chunk_probs, axis=0)[-k:].mean(axis=0)


def format_instruments(song_probs, threshold: float = CONFIDENCE_THRESHOLD) -> list[dict]:
    """Threshold, drop the stems Demucs already answers, sort by confidence."""
    instruments = []
    for index in np.argsort(-np.asarray(song_probs)):
        name = INDEX_TO_NAME[int(index)]
        probability = float(song_probs[int(index)])
        if name in DEAD_WEIGHT or probability < threshold:
            continue
        instruments.append({"name": name, "confidence": round(probability, 3)})
    return instruments


def build_timeline(chunk_probs: np.ndarray, floor: float = 0.3) -> list[dict]:
    """Per-chunk top instruments, for "what is playing at 0:30?" questions.

    Deliberately noisier than the song-level verdict — that noise is exactly why
    ``aggregate`` exists — so this reports only the few classes above ``floor``.
    """
    timeline = []
    for position, probabilities in enumerate(chunk_probs):
        top = {
            INDEX_TO_NAME[int(index)]: round(float(probabilities[index]), 2)
            for index in np.argsort(-probabilities)[:5]
            if INDEX_TO_NAME[int(index)] not in DEAD_WEIGHT
            and probabilities[index] >= floor
        }
        timeline.append({"t": position * 10, "top": top})
    return timeline


# --- backends ----------------------------------------------------------------

class Classifier:
    """Loaded weights plus the backend-specific path from audio to chunk probs.

    Construction is expensive (a 345 MB checkpoint) and per-song inference is
    cheap, so build this once per process and reuse it.
    """

    def __init__(self, backend: str, device: torch.device) -> None:
        self.backend = backend
        self.device = device
        if backend == "beats":
            self.encoder = load_backbone(
                MODEL_DIR / "beats" / "BEATs_iter3_plus_AS2M.pt", device
            )
            self.head = load_head(MODEL_DIR / HEAD_FILES[CLASSIFIER_HEAD], device)
        elif backend == "cnn":
            self.encoder = None
            self.head = Model().to(device)
            self.head.load_state_dict(
                torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
            )
            self.head.eval()
        else:
            raise ValueError(f"unknown BACKEND {backend!r}; use 'beats' or 'cnn'")

    def chunk_probabilities(self, audio_path) -> np.ndarray:
        """Audio path -> (n_chunks, 20) sigmoid probabilities."""
        if self.backend == "beats":
            waveform, _ = librosa.load(audio_path, sr=SR16, mono=True)
            chunks = to_chunks(waveform)
            outputs = []
            for start in range(0, len(chunks), 16):        # bound peak memory
                embeddings = embed_chunks(
                    self.encoder, chunks[start:start + 16], self.device
                )
                with torch.no_grad():
                    logits = self.head(torch.from_numpy(embeddings).to(self.device))
                outputs.append(torch.sigmoid(logits).cpu().numpy())
            return np.concatenate(outputs)

        mel = extract_mel_spectrogram(Path(audio_path))
        chunks = _chunk_mel(mel)
        if not chunks:
            return np.zeros((0, len(CLASS_MAP)), dtype=np.float32)
        with torch.no_grad():
            logits = self.head(torch.stack(chunks).to(self.device))
        return torch.sigmoid(logits).cpu().numpy()

    def predict(self, audio_path) -> tuple[list[dict], list[dict]]:
        """Audio path -> (song-level instruments, per-chunk timeline)."""
        probabilities = self.chunk_probabilities(audio_path)
        if len(probabilities) == 0:
            return [], []
        return (
            format_instruments(aggregate(probabilities)),
            build_timeline(probabilities),
        )


def _chunk_mel(mel) -> list[torch.Tensor]:
    """Split a (128, T) mel into (1, 128, CHUNK_FRAMES) tensors for the CNN."""
    chunks = []
    total = mel.shape[1]
    start = 0
    while start < total:
        piece = mel[:, start:start + CHUNK_FRAMES]
        width = piece.shape[1]
        if width == CHUNK_FRAMES:
            chunk = piece
        elif width >= MIN_REMAINDER_FRAMES:
            chunk = np.pad(piece, ((0, 0), (0, CHUNK_FRAMES - width)))
        else:
            break
        chunks.append(torch.from_numpy(chunk).unsqueeze(0).float())
        start += CHUNK_FRAMES
    return chunks


def load_classifier(device: torch.device | None = None,
                    backend: str = BACKEND) -> Classifier:
    """Load the configured backend once. Call this at process start."""
    return Classifier(backend, device or pick_device())


def classify(other_path, classifier: Classifier) -> list[dict]:
    """Convenience wrapper for callers that only want the instrument list."""
    instruments, _ = classifier.predict(other_path)
    return instruments
