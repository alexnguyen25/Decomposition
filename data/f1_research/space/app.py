"""Decomposition — Hugging Face Space demo.

Upload a song (or pick an example) -> Demucs stems + instrument detection
(PANNs CNN14 embeddings + trained head) + BPM/key -> players + JSON.

Deployment notes (for Alex):
- This file + requirements.txt + README.md go in the Space repo root.
- Model files expected next to app.py: Cnn14_mAP=0.431.pth, e7_head.pt,
  class-map.json (copy from data/f1_research + data/openmic/openmic-2018).
- On ZeroGPU Spaces, add `import spaces` and decorate analyze with
  @spaces.GPU(duration=120); on free CPU Basic it runs as-is (slower).
"""

import json
import time
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
from demucs.apply import apply_model
from demucs.pretrained import get_model
from panns_inference.models import Cnn14

HERE = Path(__file__).parent
SR_OUT = 44100
DEAD_WEIGHT = {"bass", "cymbals", "drums", "voice"}

# ---- load everything once per process ----------------------------------
DEMUCS = get_model("htdemucs")
DEMUCS.eval()

CNN14 = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
              mel_bins=64, fmin=50, fmax=14000, classes_num=527)
_ckpt = torch.load(HERE / "Cnn14_mAP=0.431.pth", map_location="cpu")
CNN14.load_state_dict(_ckpt["model"])
CNN14.eval()


class Head(torch.nn.Module):
    def __init__(self, d_in=2048, d_hidden=1024, p_drop=0.5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_hidden), torch.nn.BatchNorm1d(d_hidden),
            torch.nn.ReLU(), torch.nn.Dropout(p_drop),
            torch.nn.Linear(d_hidden, 20),
        )

    def forward(self, x):
        return self.net(x)


HEAD = Head()
HEAD.load_state_dict(torch.load(HERE / "e7_head.pt", map_location="cpu",
                                weights_only=True))
HEAD.eval()

with open(HERE / "class-map.json") as f:
    CLASS_MAP = json.load(f)
IDX_TO_NAME = {i: n for n, i in CLASS_MAP.items()}


def detect_instruments(other_44k: np.ndarray) -> list[dict]:
    """Other stem waveform (44.1k mono) -> [{name, confidence}] via top-3-mean."""
    y32 = librosa.resample(other_44k, orig_sr=SR_OUT, target_sr=32000)
    win = 32000 * 10
    starts = range(0, max(1, len(y32) - 32000 * 3), win)
    chunks = []
    for s in starts:
        c = y32[s:s + win]
        chunks.append(np.pad(c, (0, win - len(c))) if len(c) < win else c)
    with torch.no_grad():
        emb = CNN14(torch.from_numpy(np.stack(chunks).astype(np.float32)))[
            "embedding"]
        probs = torch.sigmoid(HEAD(emb)).numpy()
    song = np.sort(probs, axis=0)[-3:].mean(0)  # top-3-mean over chunks
    out = []
    for c in np.argsort(-song):
        name = IDX_TO_NAME[int(c)]
        if name in DEAD_WEIGHT or song[c] < 0.5:
            continue
        out.append({"name": name, "confidence": round(float(song[c]), 3)})
    return out


def analyze(audio_path: str, progress=gr.Progress()):
    t0 = time.time()
    progress(0.05, desc="Loading audio")
    wav, _ = librosa.load(audio_path, sr=SR_OUT, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])

    progress(0.15, desc="Separating stems (Demucs) — the slow part")
    with torch.no_grad():
        sources = apply_model(DEMUCS, torch.from_numpy(wav).float()
                              .unsqueeze(0), device="cpu", progress=False)[0]
    stems = {name: sources[i].mean(0).numpy()
             for i, name in enumerate(DEMUCS.sources)}

    progress(0.7, desc="Detecting instruments")
    instruments = detect_instruments(stems["other"])

    progress(0.85, desc="Estimating BPM and key")
    tempo, _ = librosa.beat.beat_track(y=stems["drums"], sr=SR_OUT)
    bpm = round(float(np.atleast_1d(tempo)[0]), 1)
    try:
        import essentia.standard as es
        key, scale, _ = es.KeyExtractor()(stems["other"].astype(np.float32))
        key_str = f"{key} {scale}"
    except Exception:  # essentia optional on Spaces
        key_str = "n/a"

    result = {
        "bpm": bpm, "key": key_str,
        "instruments": instruments,
        "elapsed_s": round(time.time() - t0, 1),
    }
    players = [(SR_OUT, stems[n]) for n in ("vocals", "drums", "bass", "other")]
    table = "\n".join(f"- **{d['name']}** — {d['confidence']:.0%}"
                      for d in instruments) or "_none above threshold_"
    return (*players, f"**BPM:** {bpm} · **Key:** {key_str}\n\n{table}",
            json.dumps(result, indent=2))


DESC = """Upload a song → get 4 separated stems (Demucs), detected instruments
(custom classifier: PANNs CNN14 embeddings + head trained on OpenMIC-2018,
test macro-F1 0.79), BPM and key. CPU processing takes ~1–4 min per song."""

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(type="filepath", label="Song (mp3/wav, ≤6 min)"),
    outputs=[
        gr.Audio(label="Vocals"), gr.Audio(label="Drums"),
        gr.Audio(label="Bass"), gr.Audio(label="Other"),
        gr.Markdown(label="Analysis"),
        gr.Code(label="JSON", language="json"),
    ],
    title="Decomposition — song → stems + instruments + BPM/key",
    description=DESC,
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
