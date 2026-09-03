"""Evaluate classifiers on real Jamendo songs with known instrument tags.

For each prepared track (see jamendo_prep.py): chunk the Demucs "other"-stem
mel into 431-frame windows (mirroring src/classification/classifier.py),
predict with several models/aggregations, and compare against MTG-Jamendo
instrument tags mapped to OpenMIC classes.

Caveat recorded in output: MTG tags are not exhaustive, so "extra" predictions
are an upper bound on false positives; "misses" on tagged instruments are real.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from experiment import (  # noqa: E402
    DEVICE, OPENMIC_DIR, PROJECT, BaselineModel, CNN4,
)

OUT = SCRATCH / "jamendo"
DEAD_WEIGHT = {2, 5, 6, 19}
KEEP = [c for c in range(20) if c not in DEAD_WEIGHT]


def chunk_mel(mel, chunk=431, min_frames=130):
    chunks, start, t = [], 0, mel.shape[1]
    while start < t:
        piece = mel[:, start:start + chunk]
        w = piece.shape[1]
        if w == chunk:
            chunks.append(piece)
        elif w >= min_frames:
            padded = np.full((mel.shape[0], chunk), -80.0, dtype=mel.dtype)
            padded[:, :w] = piece
            chunks.append(padded)
        start += chunk
    return np.stack(chunks) if chunks else None


def probs_for(model, mel, normalize):
    chunks = chunk_mel(mel)
    if chunks is None:
        return None
    xb = torch.from_numpy(chunks.astype(np.float32)).unsqueeze(1)
    if normalize:
        xb = (xb + 40.0) / 40.0
    with torch.no_grad():
        return torch.sigmoid(model(xb.to(DEVICE))).cpu().numpy()  # (C, 20)


def best_experiment(results_path):
    best = None
    for line in open(results_path):
        r = json.loads(line)
        if r["name"].startswith(("E2", "E3", "E4")):
            if best is None or (r["test_macro_f1_tuned_th"]
                                > best["test_macro_f1_tuned_th"]):
                best = r
    return best


def main():
    with open(OUT / "manifest.json") as f:
        manifest = json.load(f)
    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    name_of = {i: n for n, i in class_map.items()}

    models = {}
    baseline = BaselineModel().to(DEVICE)
    baseline.load_state_dict(torch.load(PROJECT / "models" / "classifier.pt",
                                        map_location=DEVICE, weights_only=True))
    baseline.eval()
    models["baseline"] = (baseline, False, np.full(20, 0.5))

    best = best_experiment(SCRATCH / "results.jsonl")
    if best:
        cnn = CNN4().to(DEVICE)
        cnn.load_state_dict(torch.load(SCRATCH / f"ckpt_{best['name']}.pt",
                                       map_location=DEVICE, weights_only=True))
        cnn.eval()
        models[best["name"]] = (cnn, True,
                                np.array(best["tuned_thresholds"]))
        print(f"best mel-CNN: {best['name']}", flush=True)

    report = {}
    for tid, meta in manifest.items():
        truth = set(meta["classes"])
        entry = {"tags": sorted(truth), "models": {}}
        mel_other = np.load(OUT / f"{tid}_other_mel.npy")
        mel_mix = np.load(OUT / f"{tid}_mix_mel.npy")

        for mname, (model, norm, tuned_th) in models.items():
            p_other = probs_for(model, mel_other, norm)
            p_mix = probs_for(model, mel_mix, norm)
            variants = {
                "other_max_0.5": (p_other.max(0), np.full(20, 0.5)),
                "other_mean_0.5": (p_other.mean(0), np.full(20, 0.5)),
                "other_max_tuned": (p_other.max(0), tuned_th),
                "other_mean_tuned": (p_other.mean(0), tuned_th),
                "mix_mean_tuned": (p_mix.mean(0), tuned_th),
            }
            mv = {}
            for vname, (probs, th) in variants.items():
                pred = {name_of[c] for c in KEEP if probs[c] >= th[c]}
                mv[vname] = {
                    "predicted": sorted(pred),
                    "hits": sorted(pred & truth),
                    "misses": sorted(truth - pred),
                    "extras": sorted(pred - truth),
                }
            entry["models"][mname] = mv
        report[tid] = entry
        print(f"track {tid} done", flush=True)

    # aggregate summary
    print("\n=== summary (per model/variant): tag-recall, extras/track ===")
    summary = {}
    for mname in models:
        for vname in ["other_max_0.5", "other_mean_0.5", "other_max_tuned",
                      "other_mean_tuned", "mix_mean_tuned"]:
            tot_tags = tot_hits = tot_extras = 0
            for tid in report:
                r = report[tid]["models"][mname][vname]
                tot_tags += len(report[tid]["tags"])
                tot_hits += len(r["hits"])
                tot_extras += len(r["extras"])
            rec = tot_hits / tot_tags if tot_tags else 0.0
            summary[f"{mname}/{vname}"] = {
                "tag_recall": round(rec, 3),
                "extras_per_track": round(tot_extras / len(report), 2),
            }
            print(f"{mname:>22}/{vname:<16} recall {rec:.3f} "
                  f"extras/track {tot_extras / len(report):.2f}")

    with open(SCRATCH / "jamendo_results.json", "w") as f:
        json.dump({"per_track": report, "summary": summary}, f, indent=2)
    print("JAMENDO EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
