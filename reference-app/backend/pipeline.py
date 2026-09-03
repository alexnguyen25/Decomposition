"""The analysis pipeline: song file -> stems + instruments + BPM/key + text.

Mirrors the research pipeline (src/main.py's analyze, upgraded per the
research docs): Demucs htdemucs -> classify "other" stem with BEATs head ->
BPM from drums stem -> key from other stem -> grounded LLM description.

progress_cb(fraction, stage_label) keeps the API's job status meaningful —
Demucs dominates wall time, so users need to see *why* they're waiting.
"""

import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from llm import describe
from models_util import classify_other_stem
from settings import DEAD_WEIGHT  # noqa: F401  (documented contract)

SR_NATIVE = 44100
STEM_ORDER = ("vocals", "drums", "bass", "other")

# Demucs loads once per process, same reasoning as the classifier models.
from demucs.apply import apply_model  # noqa: E402
from demucs.pretrained import get_model  # noqa: E402

DEMUCS = get_model("htdemucs")
DEMUCS.eval()


def _silence_fraction(y: np.ndarray, floor: float = 0.01) -> float:
    """Fraction of RMS frames above the floor (stem-presence heuristic,
    same frame-level approach as src/separation/stem_presence.py)."""
    rms = librosa.feature.rms(y=y)[0]
    return float((rms > floor).mean())


def _activity_envelope(y: np.ndarray, hop_s: float = 1.0) -> list[float]:
    """Loudness-over-time for one stem: one RMS value per second, normalized
    to the stem's own peak (0..1). Feeds the chat agent's get_stem_activity
    tool ("when do the vocals come in?") — relative loudness is the right
    scale for that question, absolute dB is not."""
    hop = int(SR_NATIVE * hop_s)
    rms = librosa.feature.rms(y=y, frame_length=4096, hop_length=hop)[0]
    peak = float(rms.max())
    if peak <= 0:
        return [0.0] * len(rms)
    return [round(float(v) / peak, 2) for v in rms]


def analyze(audio_path: Path, out_dir: Path, progress_cb=lambda f, s: None):
    """Run the full pipeline. Writes stem MP3s into out_dir; returns result dict."""
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    progress_cb(0.02, "Loading audio")
    wav, _ = librosa.load(audio_path, sr=SR_NATIVE, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    duration_s = wav.shape[1] / SR_NATIVE

    progress_cb(0.08, "Separating stems (Demucs) — the slow part")
    with torch.no_grad():
        sources = apply_model(DEMUCS, torch.from_numpy(wav).float()
                              .unsqueeze(0), device="cpu", progress=False)[0]
    stems = {name: sources[i].mean(0).numpy()
             for i, name in enumerate(DEMUCS.sources)}

    progress_cb(0.60, "Encoding stems")
    stem_urls = {}
    for name in STEM_ORDER:
        # MP3 (~1 MB/min) instead of WAV (~10 MB/min): browser-friendly.
        path = out_dir / f"{name}.mp3"
        sf.write(path, stems[name], SR_NATIVE, format="MP3")
        stem_urls[name] = path.name

    progress_cb(0.68, "Detecting instruments")
    other_16k = librosa.resample(stems["other"], orig_sr=SR_NATIVE,
                                 target_sr=16000)
    instruments, chunk_instruments = classify_other_stem(other_16k)

    progress_cb(0.85, "Estimating BPM and key")
    tempo, _ = librosa.beat.beat_track(y=stems["drums"], sr=SR_NATIVE)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    try:
        import essentia.standard as es
        key, scale, _ = es.KeyExtractor()(stems["other"].astype(np.float32))
        key_str = f"{key} {scale}"
    except Exception:                                # essentia is optional
        key_str = None

    presence = {name: _silence_fraction(stems[name]) >= 0.05
                for name in ("vocals", "drums", "bass")}

    result = {
        "duration_s": round(duration_s, 1),
        "bpm": bpm,
        "key": key_str,
        "presence": presence,
        "instruments": instruments,
        "stems": stem_urls,
        # time-resolved view for the chat agent's tools: per-chunk classifier
        # detections + one loudness value per second per stem.
        "timeline": {
            "chunk_s": 10,
            "instruments": chunk_instruments,
            "stem_activity": {
                "hop_s": 1.0,
                **{name: _activity_envelope(stems[name])
                   for name in STEM_ORDER},
            },
        },
    }

    progress_cb(0.92, "Writing description")
    result["description"] = describe(result)

    result["timings_s"] = {"total": round(time.time() - t0, 1)}
    progress_cb(1.0, "Done")
    return result
