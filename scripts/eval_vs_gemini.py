"""Compare this project's classifier against Gemini on the same 100 clips.

No API calls: Gemini's per-clip predictions from the July benchmark are cached
in data/f1_research/, so this re-scores them against a fresh classifier arm
computed from src/. Re-running Gemini itself needs
data/f1_research/gemini_bench.py and an API key.

Scoring is masked macro-F1 — only labels a human confirmed count, which is the
same metric used everywhere else in this project. A set-prediction model
(Gemini names instruments) and a threshold model (ours emits probabilities) are
put on equal footing by turning both into a set of names per clip.

Usage:
    python scripts/eval_vs_gemini.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.classification.classifier import CLASS_MAP, INDEX_TO_NAME
from src.classification.embedder import pick_device
from src.classification.head import load_head
from src.config import HEAD_FILES, MODEL_DIR, OPENMIC_DIR

RESEARCH = Path("data/f1_research")
ASSETS = Path(__file__).resolve().parents[1] / "src" / "classification" / "assets"
BENCH = RESEARCH / "gemini_bench_test.json"
EMBEDDINGS = RESEARCH / "beats_split01_test.npz"


def masked_scores(pred_sets: dict, y, m, keys) -> dict:
    """Per-class P/R/F1 over confirmed labels only; macro-F1.

    Copied deliberately from data/f1_research/gemini_bench.py so the arms stay
    numerically comparable to the published July table.
    """
    per_class = {}
    for name, c in CLASS_MAP.items():
        tp = fp = fn = tn = 0
        for i, key in enumerate(keys):
            if m[i, c] != 1:
                continue
            true_positive = y[i, c] >= 0.5
            predicted = name in pred_sets[key]
            tp += true_positive and predicted
            fp += (not true_positive) and predicted
            fn += true_positive and not predicted
            tn += (not true_positive) and (not predicted)
        p = float(tp / (tp + fp)) if tp + fp else 0.0
        r = float(tp / (tp + fn)) if tp + fn else 0.0
        per_class[name] = {"p": p, "r": r,
                           "f1": 2 * p * r / (p + r) if p + r else 0.0,
                           "support": int(tp + fp + fn + tn)}
    return {"per_class": per_class,
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()]))}


def main() -> None:
    if not BENCH.exists():
        sys.exit(f"missing {BENCH} — the cached Gemini benchmark is required")

    bench = json.loads(BENCH.read_text())
    keys = bench["keys"]

    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    row = {str(k): i for i, k in enumerate(npz["sample_key"])}
    index = [row[k] for k in keys]
    y, m = npz["Y_true"][index], npz["Y_mask"][index]

    # --- our arm, computed from src/ on the same clips --------------------
    device = pick_device()
    cache = np.load(EMBEDDINGS, allow_pickle=True)
    embedding_row = {str(k): i for i, k in enumerate(cache["keys"])}
    embeddings = cache["embeddings"][[embedding_row[k] for k in keys]]
    head = load_head(MODEL_DIR / HEAD_FILES["e10"], device)
    with torch.no_grad():
        probabilities = torch.sigmoid(
            head(torch.from_numpy(embeddings).to(device))
        ).cpu().numpy()

    thresholds = np.asarray(
        json.loads((ASSETS / "thresholds.json").read_text())["e10_tuned_thresholds"],
        dtype=np.float32,
    )

    arms = {
        "BEATs + head @0.5": {
            k: {INDEX_TO_NAME[c] for c in range(20) if probabilities[i, c] >= 0.5}
            for i, k in enumerate(keys)},
        "BEATs + head @tuned": {
            k: {INDEX_TO_NAME[c] for c in range(20)
                if probabilities[i, c] >= thresholds[c]}
            for i, k in enumerate(keys)},
    }

    # --- cached arms from the July benchmark ------------------------------
    for model in ("gemini-flash-latest", "gemini-pro-latest"):
        cache_path = RESEARCH / f"gemini_cache_{model}.json"
        if cache_path.exists():
            predictions = json.loads(cache_path.read_text())
            if all(k in predictions for k in keys):
                arms[model] = {k: set(predictions[k]) for k in keys}

    print(f"{len(keys)} OpenMIC test clips · masked macro-F1 "
          f"(confirmed labels only)\n")
    scored = {name: masked_scores(preds, y, m, keys) for name, preds in arms.items()}
    for name, score in sorted(scored.items(), key=lambda kv: -kv[1]["macro_f1"]):
        published = bench.get(name, {}).get("macro_f1")
        suffix = f"   (July run: {published:.4f})" if published else ""
        print(f"  {name:24s} {score['macro_f1']:.4f}{suffix}")

    for name in ("E7_panns@tuned", "E4_cnn@0.5"):
        if name in bench:
            print(f"  {name:24s} {bench[name]['macro_f1']:.4f}   (cached, July)")

    out = RESEARCH / "gemini_bench_beats.json"
    out.write_text(json.dumps(
        {"n_clips": len(keys),
         "arms": {k: {"macro_f1": v["macro_f1"], "per_class": v["per_class"]}
                  for k, v in scored.items()}}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
