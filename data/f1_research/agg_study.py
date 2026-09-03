"""Aggregation study on real Jamendo songs: how to pool chunk probabilities
over a full track without max-pooling's false-positive explosion.

Models: E4 (mel CNN) on other-stem mels; E7 (PANNs head) on other-stem audio.
Aggregators: max, mean, top3-mean, top25%-mean.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "pylibs"))
from experiment import DEVICE, OPENMIC_DIR, CNN4  # noqa: E402
from jamendo_eval import chunk_mel, DEAD_WEIGHT, KEEP  # noqa: E402
from panns_head import MLPHead  # noqa: E402
from panns_inference.models import Cnn14  # noqa: E402

OUT = SCRATCH / "jamendo"
SR32 = 32000
WIN = SR32 * 10


def e4_chunk_probs(mel):
    model = e4_chunk_probs.model
    chunks = chunk_mel(mel)
    xb = (torch.from_numpy(chunks.astype(np.float32)).unsqueeze(1) + 40.0) / 40.0
    with torch.no_grad():
        return torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()


def e7_chunk_probs(wave):
    cnn14, head = e7_chunk_probs.models
    pieces = []
    for s in range(0, len(wave) - SR32 * 3, WIN):
        w = wave[s:s + WIN]
        if len(w) < WIN:
            w = np.pad(w, (0, WIN - len(w)))
        pieces.append(w)
    x = torch.from_numpy(np.stack(pieces).astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        emb = cnn14(x)["embedding"]
        return torch.sigmoid(head(emb)).cpu().numpy()


AGGS = {
    "max": lambda p: p.max(0),
    "mean": lambda p: p.mean(0),
    "top3mean": lambda p: np.sort(p, axis=0)[-3:].mean(0),
    "top25pct": lambda p: np.sort(p, axis=0)[-max(1, int(np.ceil(len(p) * .25))):].mean(0),
}


def main():
    with open(OUT / "manifest.json") as f:
        manifest = json.load(f)
    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    name_of = {i: n for n, i in class_map.items()}

    e4 = CNN4().to(DEVICE)
    e4.load_state_dict(torch.load(SCRATCH / "ckpt_E4_cnn4_specaug_posweight.pt",
                                  map_location=DEVICE, weights_only=True))
    e4.eval()
    e4_chunk_probs.model = e4

    cnn14 = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ckpt = torch.load(SCRATCH / "Cnn14_mAP=0.431.pth", map_location="cpu")
    cnn14.load_state_dict(ckpt["model"])
    cnn14 = cnn14.to(DEVICE).eval()
    head = MLPHead().to(DEVICE)
    head.load_state_dict(torch.load(SCRATCH / "ckpt_E7_panns_head.pt",
                                    map_location=DEVICE, weights_only=True))
    head.eval()
    e7_chunk_probs.models = (cnn14, head)

    counts = {}
    for tid, meta in manifest.items():
        truth = set(meta["classes"])
        p_e4 = e4_chunk_probs(np.load(OUT / f"{tid}_other_mel.npy"))
        wave = np.load(OUT / f"{tid}_other_32k.npy").astype(np.float32)
        p_e7 = e7_chunk_probs(wave)

        for mname, probs in [("E4", p_e4), ("E7_panns", p_e7)]:
            for aname, agg in AGGS.items():
                song = agg(probs)
                pred = {name_of[c] for c in KEEP if song[c] >= 0.5}
                k = f"{mname}/{aname}"
                c = counts.setdefault(k, {"tags": 0, "hits": 0, "extras": 0})
                c["tags"] += len(truth)
                c["hits"] += len(pred & truth)
                c["extras"] += len(pred - truth)
        print(f"{tid} done", flush=True)

    print(f"\n{'model/agg':<22} {'recall':>7} {'extras/track':>13}")
    n = len(manifest)
    for k, c in counts.items():
        print(f"{k:<22} {c['hits'] / c['tags']:>7.3f} {c['extras'] / n:>13.2f}")

    with open(SCRATCH / "agg_study.json", "w") as f:
        json.dump({k: {"recall": c["hits"] / c["tags"],
                       "extras_per_track": c["extras"] / n}
                   for k, c in counts.items()}, f, indent=2)
    print("AGG STUDY DONE", flush=True)


if __name__ == "__main__":
    main()
