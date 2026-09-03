"""Reproduce the published OpenMIC-2018 macro-F1 using only src/ code.

This is the regression gate for the classifier: if a change to the embedder,
the head, or the protocol breaks the numbers in docs/research/, this script
says so. It uses cached BEATs embeddings when they exist and otherwise embeds
the raw audio itself, so it works from a clean checkout of the dataset.

Usage:
    python scripts/eval_openmic.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.embedder import (
    embed_chunks,
    load_backbone,
    pick_device,
)
from src.classification.evaluate import compute_metrics
from src.classification.head import load_head
from src.config import (
    CHUNK_SAMPLES,
    CLASSIFIER_HEAD,
    HEAD_FILES,
    MODEL_DIR,
    OPENMIC_DIR,
    SR16,
)

ASSETS = Path(__file__).resolve().parents[1] / "src" / "classification" / "assets"
CACHED_EMBEDDINGS = Path("data/f1_research/beats_split01_test.npz")


def load_labels(keys: list[str]):
    """Confirmed-label matrix and mask for the given OpenMIC sample keys."""
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    row = {str(k): i for i, k in enumerate(npz["sample_key"])}
    index = [row[k] for k in keys]
    labels = (npz["Y_true"][index] >= 0.5).astype(np.int8)
    masks = npz["Y_mask"][index].astype(np.int8)
    return labels, masks


def embeddings_for_test_split(device):
    """(embeddings, keys) — from cache when present, else computed from audio."""
    if CACHED_EMBEDDINGS.exists():
        cache = np.load(CACHED_EMBEDDINGS, allow_pickle=True)
        print(f"using cached embeddings: {CACHED_EMBEDDINGS}")
        return cache["embeddings"], [str(k) for k in cache["keys"]]

    print("no embedding cache — embedding raw audio with src/ (slow)")
    import librosa
    with open(OPENMIC_DIR / "partitions" / "split01_test.csv") as f:
        keys = [line.strip() for line in f if line.strip()]
    backbone = load_backbone(MODEL_DIR / "beats" / "BEATs_iter3_plus_AS2M.pt", device)
    out = []
    for start in range(0, len(keys), 16):
        batch = []
        for key in keys[start:start + 16]:
            y, _ = librosa.load(
                OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg", sr=SR16, mono=True
            )
            clip = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
            clip[:min(len(y), CHUNK_SAMPLES)] = y[:CHUNK_SAMPLES]
            batch.append(clip)
        out.append(embed_chunks(backbone, np.stack(batch), device))
        if start % 800 == 0:
            print(f"  {start}/{len(keys)}", flush=True)
    return np.concatenate(out), keys


def verify_embedder_matches_cache(device) -> None:
    """Guard: src/'s vendored BEATs must produce the cache's exact embeddings.

    Without this, the head could be evaluated against embeddings that src/ can
    no longer produce, and the reported number would describe research code
    rather than shipped code.
    """
    if not CACHED_EMBEDDINGS.exists():
        print("skip embedder check (no cache to compare against)")
        return
    import librosa
    cache = np.load(CACHED_EMBEDDINGS, allow_pickle=True)
    keys = [str(k) for k in cache["keys"]][:24]
    reference = cache["embeddings"][:24]
    backbone = load_backbone(MODEL_DIR / "beats" / "BEATs_iter3_plus_AS2M.pt", device)
    batch = []
    for key in keys:
        y, _ = librosa.load(
            OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg", sr=SR16, mono=True
        )
        clip = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
        clip[:min(len(y), CHUNK_SAMPLES)] = y[:CHUNK_SAMPLES]
        batch.append(clip)
    produced = embed_chunks(backbone, np.stack(batch), device)
    difference = float(np.abs(produced - reference).max())
    print(f"embedder vs cache on {len(keys)} clips: max abs diff {difference:.2e}")
    assert difference < 1e-4, "src/ embedder no longer reproduces the cached embeddings"


def main() -> None:
    device = pick_device()
    print(f"device: {device} | head: {CLASSIFIER_HEAD}\n")

    verify_embedder_matches_cache(device)

    embeddings, keys = embeddings_for_test_split(device)
    labels, masks = load_labels(keys)
    head = load_head(MODEL_DIR / HEAD_FILES[CLASSIFIER_HEAD], device)

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(embeddings), 1024):
            batch = torch.from_numpy(embeddings[start:start + 1024]).to(device)
            probabilities.append(torch.sigmoid(head(batch)).cpu().numpy())
    probabilities = np.concatenate(probabilities)

    published = json.loads((ASSETS / "thresholds.json").read_text())
    thresholds = np.asarray(published["e10_tuned_thresholds"], dtype=np.float32)

    at_half = compute_metrics((probabilities >= 0.5).astype(np.int8), labels, masks)
    tuned = compute_metrics(
        (probabilities >= thresholds[None, :]).astype(np.int8), labels, masks
    )

    print(f"\n  test clips: {len(embeddings)}")
    print(f"  macro-F1 @0.5    {at_half['macro_f1']:.4f}   "
          f"(published {published['published_macro_f1_at_0.5']:.4f})")
    print(f"  macro-F1 tuned   {tuned['macro_f1']:.4f}   "
          f"(published {published['published_macro_f1_tuned']:.4f})")

    drift = abs(tuned["macro_f1"] - published["published_macro_f1_tuned"])
    print("\nRESULT:", "REPRODUCED" if drift < 1e-4 else f"DRIFTED by {drift:.4f}")
    sys.exit(0 if drift < 1e-4 else 1)


if __name__ == "__main__":
    main()
