"""Score src/'s classifier on real songs with known instrumentation.

OpenMIC clips are 10 s of full-mix audio; the product sees a Demucs "other"
stem from a whole song. That gap is why a good OpenMIC number does not by
itself mean a good product, so this evaluates on the deployment distribution.

Ground truth is MTG-Jamendo instrument tags (weak and incomplete — an untagged
instrument that is genuinely present counts here as a false positive, so treat
precision as a lower bound).

Usage:
    python scripts/eval_real_songs.py            # default backend
    BACKEND=cnn python scripts/eval_real_songs.py
"""

import json
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.classifier import load_classifier
from src.config import SR16

JAMENDO = Path("data/f1_research/jamendo")


def scored(predicted: set[str], truth: set[str]) -> tuple[float, float, float]:
    hits = len(predicted & truth)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def main() -> None:
    manifest_path = JAMENDO / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path} — see docs/research/ for how it was built")
    manifest = json.loads(manifest_path.read_text())

    classifier = load_classifier()
    print(f"backend: {classifier.backend} on {classifier.device}\n")

    totals = {"p": 0.0, "r": 0.0, "f1": 0.0, "predicted": 0}
    tags = hits = 0
    with tempfile.TemporaryDirectory() as tmp:
        for track_id, meta in manifest.items():
            stem = JAMENDO / f"{track_id}_other_32k.npy"
            if not stem.exists():
                print(f"  skip {track_id} (no cached stem)")
                continue
            waveform = librosa.resample(
                np.load(stem).astype(np.float32), orig_sr=32000, target_sr=SR16
            )
            wav_path = Path(tmp) / f"{track_id}.wav"
            sf.write(wav_path, data=waveform, samplerate=SR16)

            instruments, _ = classifier.predict(wav_path)
            predicted = {i["name"] for i in instruments}
            truth = set(meta["classes"])
            precision, recall, f1 = scored(predicted, truth)

            totals["p"] += precision
            totals["r"] += recall
            totals["f1"] += f1
            totals["predicted"] += len(predicted)
            tags += len(truth)
            hits += len(predicted & truth)

            print(f"  {track_id}: P {precision:.2f} R {recall:.2f} F1 {f1:.2f}"
                  f"  | missed {sorted(truth - predicted)}"
                  f"  | extra {sorted(predicted - truth)}")

    n = len(manifest)
    print(f"\n  tracks {n}")
    print(f"  mean precision   {totals['p'] / n:.3f}")
    print(f"  mean recall      {totals['r'] / n:.3f}")
    print(f"  mean F1          {totals['f1'] / n:.3f}")
    print(f"  micro recall     {hits / tags:.3f}")
    print(f"  predictions/track {totals['predicted'] / n:.1f}")


if __name__ == "__main__":
    main()
