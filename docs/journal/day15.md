# Handoff — Decomposition
**After session:** `evaluate.py` fully designed and written; first eval run kicked off on Colab, currently stalled/slow — investigation paused
**Next session focus:** Diagnose the stuck eval run, get real per-class metrics, then move to `classifier.py`

---

## Project Overview

**What it is:** An audio intelligence pipeline that takes a music file and outputs 4 separated stems (vocals, drums, bass, other), instrument labels for the "other" stem, BPM, and key — as structured JSON.

**Pipeline:** Preprocessing (validate/normalize) → Demucs source separation → mel spectrogram feature extraction → custom CNN classifier (trained on OpenMIC-2018) → structured output.

**User:** Alex, rising sophomore at Northeastern University, targeting FAANG/top tech co-ops for Spring/Summer 2027. This is the primary portfolio project.

**Claude's role: researcher and tutor only — never write code for Alex.** Socratic method — ask guiding questions, let Alex reason to answers, correct precisely when wrong, use interactive visualizers for spatial/mechanical concepts. Alex owns all architecture decisions, tensor shape reasoning, and writing the actual implementation code. Library/doc lookup is the one narrow exception. **This session included two separate instances of Alex pasting in a pre-written/AI-generated implementation instead of writing it themselves — both times, Claude declined to review it as if Alex had written it, asked directly whether it was actually Alex's work, and the two rebuilt the same content from scratch via the normal Socratic process instead.** Worth staying alert to this pattern continuing.

**User preference:** Never push or commit anything without permission.

---

## Current Project State

**Built, verified, and reviewed:**
- `src/config.py` — environment-aware (Colab vs. local), `TRAIN_PARTITION = "split01_train.csv"`, `TEST_PARTITION = "split01_test.csv"` (both confirmed present this session — no new config needed for `evaluate.py`)
- `src/preprocessing/validation.py`
- `src/preprocessing/processing.py`
- `src/separation/separation.py`
- `src/feature_extraction/feature_extraction.py`
- `src/classification/dataset.py`
- `src/classification/model.py`
- `src/classification/train.py`
- `precompute_mel.py` (repo root)
- Colab training notebook (5 cells)
- **`src/classification/evaluate.py` ✅ NEW — fully written and reviewed this session** (full breakdown below)

**Trained model exists:** `classifier.pt` at `/content/drive/MyDrive/Decomposition/models/classifier.pt`, 25 epochs, loss 0.667 → 0.465 (plateaued). Real per-class performance still unknown — this is exactly what `evaluate.py` exists to answer, and the first run hasn't finished yet (see "Open Issue" below).

**Not yet built:**
- `classifier.py` ← next up, once eval numbers are in hand
- `main.py` (top-level pipeline wiring)
- Full-stack web app (post-MVP)
- Gemini benchmark comparison (post-MVP, stretch goal)

---

## `evaluate.py` — Final Implementation (locked, reviewed, correct)

Five plain functions, same pattern as `train.py` (no unnecessary classes, thin `main()`):

```python
"""Evaluate the trained OpenMIC classifier on the test partition."""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.classification.dataset import OpenMICDataset
from src.classification.model import Model
from src.config import CACHE_DIR, CHECKPOINT_PATH, OPENMIC_DIR, TEST_PARTITION


def setup() -> tuple[DataLoader, Model, torch.device]:
    """Create the test DataLoader and load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = OpenMICDataset(OPENMIC_DIR, TEST_PARTITION, CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = Model().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return dataloader, model, device


def run_inference(
    dataloader: DataLoader,
    model: Model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return test-set predictions, labels, and masks."""
    prediction_batches = []
    label_batches = []
    mask_batches = []

    with torch.no_grad():
        for spec, labels, label_mask in dataloader:
            logits = model(spec.to(device))
            predictions = (torch.sigmoid(logits) >= 0.5).cpu()

            prediction_batches.append(predictions)
            label_batches.append(labels)
            mask_batches.append(label_mask)

    predictions = torch.cat(prediction_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()
    masks = torch.cat(mask_batches, dim=0).numpy()

    return predictions, labels, masks


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray,
) -> dict:
    """Compute mask-aware per-class metrics and macro-F1."""
    results = []
    num_classes = predictions.shape[1]

    for c in range(num_classes):
        confirmed = masks[:, c] == 1
        y_pred = predictions[confirmed, c]
        y_true = labels[confirmed, c]

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results.append({
            "class_index": c,
            "n_confirmed": int(confirmed.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    f1_scores = [r["f1"] for r in results]
    macro_f1 = float(np.mean(f1_scores))
    return {"classes": results, "macro_f1": macro_f1}


def print_report(metrics: dict) -> None:
    """Print the per-class metrics table and macro-F1 summary."""
    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    index_to_name = {index: name for name, index in class_map.items()}

    print(f"{'instrument':<20} {'confirmed':>9} {'precision':>9} {'recall':>9} {'f1':>9}")
    print("-" * 60)

    for r in metrics["classes"]:
        name = index_to_name[r["class_index"]]
        print(
            f"{name:<20} {r['n_confirmed']:>9} "
            f"{r['precision']:>9.3f} {r['recall']:>9.3f} {r['f1']:>9.3f}"
        )

    print("-" * 60)
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")


def main():
    """Run the complete evaluation workflow."""
    dataloader, model, device = setup()
    predictions, labels, masks = run_inference(dataloader, model, device)
    metrics = compute_metrics(predictions, labels, masks)
    print_report(metrics)


if __name__ == "__main__":
    main()
```

### Key design points to remember

**Five functions, not fewer** — `setup()`, `run_inference()`, `compute_metrics()`, `print_report()`, `main()`. Deliberately split so `compute_metrics()` stays pure (numbers in, numbers out, no I/O) and reusable — e.g., for a future side-by-side comparison of two checkpoints, or a unit test on the P/R/F1 math, neither of which would work cleanly if print statements were baked into the metrics function.

**`torch.no_grad()`** wraps the entire inference loop. Rationale worked through explicitly: PyTorch builds a computation graph for every tensor op by default (so gradients *could* be computed later), regardless of whether `.backward()` is ever called. Since evaluation never calls `.backward()`, that bookkeeping is pure overhead — `no_grad()` tells PyTorch to skip it.

**Sigmoid + threshold, in that order** — `torch.sigmoid(logits) >= 0.5`. Sigmoid squashes each of the 20 raw logits independently to a 0–1 probability (independent per-class, unlike softmax which forces a sum-to-1 "pick one" distribution — wrong fit for multi-label). The `0.5` threshold is the same cutoff already used elsewhere in the pipeline (`dataset.py`'s `Y_true` binarization) — consistent convention throughout.

**GPU/CPU device discipline, reasoned through in detail this session:**
- `spec.to(device)` — moves input to GPU right before the forward pass, since the model lives there
- `predictions = (...).cpu()` — moves the *result* back to CPU immediately after computing it, so GPU memory isn't held onto for data that's done being used on GPU
- `labels` and `label_mask` **never get a `.cpu()` call** — they come straight out of `OpenMICDataset.__getitem__` as CPU tensors and never move to GPU in the first place, so there's nothing to move back. This was explicitly reasoned through, not assumed — the fix for a naive implementation isn't "always call `.cpu()` on everything," it's understanding which tensors actually crossed to GPU in the first place.
- Rationale for per-batch `.cpu()` rather than accumulating GPU tensors and converting once at the end: holding every batch's tensors in VRAM until the full loop finishes is a classic OOM risk pattern, even though at this dataset's scale (~5,100 clips × 20 classes) it wouldn't actually run out of memory. Correct pattern regardless of scale, not a premature optimization.

**Accumulation mechanism** — plain Python lists (`prediction_batches`, `label_batches`, `mask_batches`), one append per batch, `torch.cat(list, dim=0)` at the end (concatenating along the *existing* batch dimension, not stacking a new one), `.numpy()` once after concatenation. Confirmed explicitly: `torch.cat` requires every dimension except `dim=0` to match across tensors, but `dim=0` itself can vary freely — this is why an uneven final batch (e.g. 5 clips instead of 32) causes no issue.

**Mask-based filtering, per class, done correctly** — `confirmed = masks[:, c] == 1` isolates one instrument column at a time and filters to only clips where that specific instrument's label was actually confirmed (as opposed to the training-time masked loss, which zeroed out unconfirmed contributions within one combined scalar rather than excluding rows). `masks` dtype is `float32` (not bool/int) — confirmed this doesn't cause an equality-comparison problem since mask values are exactly `1.0`/`0.0`, never fractional.

**Precision / recall / F1 hand-rolled, not scikit-learn** — real design decision made this session, not a default. Reasoning: the formulas are three lines of arithmetic (`tp/(tp+fp)`, `tp/(tp+fn)`, harmonic mean), genuinely simple to verify by hand, and the project's dependency list is otherwise lean (`librosa`, `soundfile`, `torch`). Explicitly weighed against "avoid reinventing the wheel" — concluded that phrase applies to genuinely hard/error-prone reimplementations (e.g. Demucs), not three formulas simple enough to derive from TP/FP/FN counts. Also considered from an interview-defensibility angle: being able to explain the exact math beats citing a library call for something this simple.

**Zero-division guards on all three metrics** — `if (tp + fp) > 0 else 0.0` style guards on precision, recall, and F1 independently. Not a theoretical edge case here (unlike the batch-level masked-loss div-by-zero in `train.py`, which was accepted as astronomically unlikely) — with 20 classes of varying rarity, some genuinely can hit zero predicted positives or zero confirmed true positives in the test set.

**`n_confirmed` included per class** — lets a `0.0` F1 be interpreted correctly later (distinguishes "this instrument was never confirmed in the test set" from "confirmed but the model gets it wrong").

**Macro-F1 computed inside `compute_metrics()`, not in `print_report()` or `main()`** — explicitly reasoned through: averaging a list of floats is arithmetic, not presentation, so it belongs with the other math even though it's a summarization step layered on top of the per-class loop rather than part of the loop itself.

**`class-map.json` name lookup** — loaded and index-flipped inside `print_report()` specifically (not `compute_metrics()`), keeping the metrics function's contract as "numbers in, numbers out" and letting table formatting change independently of the math. `class_map` is `name → index`; flipped once via dict comprehension (`{index: name for name, index in class_map.items()}`) rather than re-searching the original dict per lookup.

**`map_location=device` on `torch.load`** — necessary because a checkpoint saved on a CUDA device stores that device placement; without `map_location`, loading a GPU-trained checkpoint on a CPU-only machine throws `RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False`. Confirmed this is the literal error message, not just "it would error somehow."

**Model-to-device / load_state_dict ordering** — confirmed either order (`.to(device)` before or after `load_state_dict()`) works safely, since `load_state_dict()` copies values *into* existing parameter tensors rather than replacing them — as long as `.to(device)` happens at some point before running inference.

### Compute requirement, explicitly reasoned through

Evaluation is inference-only — no backward pass, no optimizer step, tiny CNN, ~5,100 forward passes at batch size 32 (~160 batches). This should be **seconds to low minutes on CPU alone** — GPU is a nice-to-have for speed, not a hard requirement the way it was for training. The actual reason to run this on Colab right now isn't compute — it's **file access**: `classifier.pt` and the precomputed mel cache both currently live in Google Drive from the training session, and haven't been downloaded locally yet. Running on Colab (where both are already reachable) is the path of least friction for this first run; running locally later would require downloading both first.

---

## Open Issue — Eval Run Stalled/Slow, Not Yet Resolved

First real run of `evaluate.py` was kicked off on Colab this session. `setup()` completed and printed correctly:
```
Using device: cuda
```
(confirming checkpoint path resolved, test partition loaded, device detected). But the run then sat for **20+ minutes** inside `run_inference()` with no further output — far beyond the expected "seconds to low minutes," especially running on `cuda`, which was a red flag rather than just "slow."

**Suspected cause (Alex's working theory, not yet confirmed): Google Drive I/O**, most likely one of:
1. **Cache miss on the test partition** — if `precompute_mel.main()` was never actually run against `split01_test.csv`'s clips (Day 14's session ran it, but this should be double-checked, not assumed), `OpenMICDataset.__getitem__` would silently fall back to live mel extraction (full Librosa computation, possibly Essentia key extraction too) for every one of ~5,100 clips, reading raw audio over Drive-mounted I/O — a dramatically slower path than reading cached `.npy` files.
2. Slow Drive-mounted file I/O in general, even on cache hits, if Drive is behaving poorly this session.

**Not yet checked, but should be the first move next session:**
- Whether the run is still actively progressing (GPU utilization via `nvidia-smi`, or any Colab activity indicator) versus fully frozen at 0% — this distinguishes "genuinely slow" from "hung."
- Whether `CACHE_DIR` actually contains `.npy` files corresponding to `split01_test.csv`'s sample keys — i.e., confirm the test-partition cache actually exists and isn't just assumed from the Day 14 journal entry.
- If cache is confirmed missing for the test partition: run `precompute_mel.main()` against `split01_test.csv` explicitly before re-attempting `evaluate.py`.

**Decision made this session:** stop investigating for now, let the current run keep going in the background, revisit next session with the specific diagnostic checks above rather than guessing further.

---

## Immediate Next Steps

1. **Diagnose and resolve the stalled eval run** — check whether it finished or is still hung, confirm/deny the cache-miss theory using the checks listed above, re-run cleanly once the cause is fixed.
2. **Get real per-class P/R/F1 numbers** — this is the actual deliverable of this whole stage. Once in hand: look specifically at whether the "dead weight" classes (voice, drums, bass, cymbals — see Day 14's distribution-shift resolution) score suspiciously differently than the other 16, since this test set is full mixes (not Demucs-separated stems) and all 20 classes are valid labels in this specific context.
3. **Move to `classifier.py`** once real eval numbers exist — chunked inference on real songs, cross-chunk max-pool aggregation, the "is this stem silent" energy-threshold check for vocals/drums/bass (flagged since Day 14, not yet built), and the ~16-class-only reporting logic at the product layer.

---

## Remaining Build Order

| Stage | Status |
|---|---|
| `config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py`, `train.py` | ✅ Done |
| Colab training run | ✅ Done — `classifier.pt` saved to Drive |
| `evaluate.py` (code) | ✅ Done (this session) |
| `evaluate.py` (first successful run + real numbers) | ← **Blocked, in progress** — stalled run under investigation |
| `classifier.py` | Not started — next after eval numbers land |
| `main.py` (pipeline wiring) | Not started |
| Full-stack web app | Post-MVP |
| Gemini benchmark comparison | Post-MVP, stretch goal / portfolio talking point |

---

## Process Notes for Next Session

- Continue the file-by-file pattern: state the file's job in one sentence, sketch function/class skeleton only, work through edge cases via guided questions, Claude supplies distilled library references as needed, Alex writes the implementation, Claude reviews after.
- **This session, twice, Alex pasted in a fully-written implementation (once framed as a "study" document with embedded full code, once as a direct code paste) instead of writing it via the normal process.** Both times Claude declined to treat it as reviewable work-product and the two rebuilt the same content from scratch through the standard question-by-question process instead. The final `evaluate.py` in this handoff **was** built that way — every line reasoned through, questioned, and written by Alex directly in chat. Worth staying alert for this pattern recurring, especially under time pressure.
- A `torch.no_grad()` / GPU-memory discipline conversation happened in real depth this session (per-batch `.cpu()` vs. accumulate-then-convert-once) — Alex reasoned through the VRAM/OOM angle correctly and independently once prompted, this is solid understanding, not memorized.
- Keep enforcing: Claude never writes implementation code, even under time pressure — the library-lookup exception is for genuine one-off doc lookups only.
- Watch the `sys.modules` import-caching trap (Day 14) if adding cells to the existing notebook for this diagnostic work — `config` must not get imported before Drive is mounted in a given Colab session, or paths lock in stale.
