# Handoff — Decomposition
**After session:** train.py build (post Day 12)
**Next session focus:** Colab training run → evaluate.py

---

## Project Overview

**What it is:** An audio intelligence pipeline that takes a music file and outputs 4 separated stems (vocals, drums, bass, other), instrument labels for the "other" stem, BPM, and key — as structured JSON.

**Pipeline:** Preprocessing (validate/normalize) → Demucs source separation → mel spectrogram feature extraction → custom CNN classifier (trained on OpenMIC-2018) → structured output.

**User:** Alex, rising sophomore at Northeastern University, targeting FAANG/top tech co-ops for Spring/Summer 2027. This is the primary portfolio project.

**Claude's role: researcher and tutor only — never write code for Alex.** Socratic method — ask guiding questions, let Alex reason to answers, correct precisely when wrong, use interactive visualizers for spatial/mechanical concepts. Alex explicitly does not want to independently research raw PyTorch docs — Claude looks up and distills exact library function signatures/behavior, but Alex still owns all architecture decisions, tensor shape reasoning, and writing the actual implementation code. This is a narrow, deliberate exception to "Claude doesn't write code" — it applies only to doc/library lookup, not to design or implementation.

**User preference:** Never push or commit anything without permission.

---

## Current Project State

**Built, verified, and reviewed:**
- `src/config.py`
- `src/preprocessing/validation.py`
- `src/preprocessing/processing.py`
- `src/separation/separation.py`
- `src/feature_extraction/feature_extraction.py`
- `src/classification/dataset.py`
- `src/classification/model.py`
- `src/classification/train.py` ✅ **NEW — just completed this session**

**Not yet built:**
- `evaluate.py` ← next up (after the Colab training run)
- `classifier.py`
- `main.py` (top-level pipeline wiring)
- Full-stack web app (post-MVP)
- Gemini benchmark comparison (post-MVP, stretch goal)

**Immediate next physical step:** Alex runs training on Google Colab (L4 GPU) using the precomputed mel cache. This session's work (`train.py`) was completed under a stated time crunch, so a few small items were consciously deferred rather than solved — see "Flagged, Not Yet Resolved" below.

---

## train.py — Final Implementation (locked, reviewed, correct)

```python
from torch.utils.data import DataLoader
from src.config import NUM_EPOCHS, OPENMIC_DIR, CACHE_DIR, TRAIN_PARTITION
from src.classification.dataset import OpenMICDataset
from src.classification.model import Model
from pathlib import Path

import torch
from torch import nn

CHECKPOINT_PATH = Path("models/classifier.pt")

def main():
    dataloader, model, optimizer, loss_fn = setup()
    trained_model = train(dataloader, model, optimizer, loss_fn, NUM_EPOCHS)
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trained_model.state_dict(), CHECKPOINT_PATH)

def setup():
    dataset = OpenMICDataset(OPENMIC_DIR, TRAIN_PARTITION, CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    return dataloader, model, optimizer, loss_fn

def train(dataloader, model, optimizer, loss_fn, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            spec, labels, label_mask = batch

            optimizer.zero_grad()

            logits = model(spec)

            per_element = loss_fn(logits, labels)
            loss = (per_element * label_mask).sum() / label_mask.sum()
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
        print(total_loss / len(dataloader))

    return model
```

### Key design points to remember

**Three plain functions, not a class** — `setup()`, `train()`, `main()`. No `__init__`/`self` anywhere; this isn't a class because there's no need to instantiate multiple training runs with persistent state. This was explicitly worked through and decided — Alex initially reached for class patterns out of habit and self-corrected.

**Masked loss formula** (the core mechanical piece of this file):
```python
per_element = loss_fn(logits, labels)              # (B, 20), reduction='none'
loss = (per_element * label_mask).sum() / label_mask.sum()   # scalar
```
- `label_mask` is 1s/0s — elementwise multiply zeros out unconfirmed labels' contribution
- Numerator: sum of the masked (real) losses only — zeros contribute nothing to a sum
- Denominator: `label_mask.sum()` counts how many labels were actually confirmed (summing 1s/0s = count of 1s)
- **Div-by-zero edge case**: theoretically possible if every label in a batch is unconfirmed, but with 20 classes × 32 clips = 640 label slots per batch and OpenMIC clips typically having several confirmed labels each, this is astronomically unlikely. Deliberately **not** guarded against with branching logic — just worth a comment if revisited. Do not add defensive code here without a real reason.

**Training loop order**: `zero_grad()` → forward pass → compute loss → `backward()` → `optimizer.step()`. Reasoning: `.backward()` *accumulates* gradients rather than overwriting them, so gradients must be cleared before each batch's backward pass. Convention is `zero_grad()` at the *start* of each iteration (not the end) — functionally equivalent, but less error-prone if the loop is edited later.

**Optimizer: Adam**, `torch.optim.Adam(model.parameters(), lr=0.001)`. Chosen for faster convergence and lower tuning overhead on a first training run — **not** because of dataset size (this reasoning was explicitly corrected mid-session; "large dataset" is not a valid justification for Adam over SGD and wouldn't hold up to a follow-up question in an interview). `lr=0.001` is the standard Adam default.

**DataLoader mechanics** — confirmed and correctly reasoned through by Alex: `DataLoader`'s default collate behavior stacks each *position* of the per-sample tuple across the batch dimension. So `dataset[i]` returning `(spec, labels, label_mask)` becomes, after batching, `spec: (32, 1, 128, 431)`, `labels: (32, 20)`, `label_mask: (32, 20)` — meaning `spec, labels, label_mask = batch` correctly unpacks three already-stacked tensors, not a list of 32 separate tuples.

**Epoch-level loss tracking** — went through several iterations before landing correctly:
- First attempt: printed `loss.item()` inside the batch loop (too noisy, one print per ~466 batches/epoch)
- Second attempt: moved print outside the batch loop but still referenced `loss` directly — bug, since `loss` gets overwritten every batch iteration, so this only ever showed the *last* batch's loss, not the epoch's
- Final/correct: accumulate `total_loss += loss.item()` inside the batch loop, then after the loop divide by `len(dataloader)` and print
- **`len(dataloader)` returns the number of *batches*** (≈466 for ~14,900 clips at batch_size=32), not the number of individual clips — this was explicitly reasoned through and confirmed. `total_loss / len(dataloader)` = mean loss per batch across the epoch, a clean epoch-over-epoch signal.

**Checkpoint pattern**:
```python
torch.save(model.state_dict(), CHECKPOINT_PATH)   # save
# later, in evaluate.py / classifier.py:
model = Model()
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()
```
`state_dict()` is just numbers (weight tensors labeled by layer name) — it has no architecture of its own. You must build `Model()` first (giving it random weights + structure), then load the saved numbers into that structure. `.eval()` switches to inference mode (doesn't affect this architecture directly since there's no Dropout/BatchNorm, but is a correct habit to build now).

**Return-value design** — `train()` deliberately only returns the trained `model`, not the state_dict, and does not handle saving itself. `main()` owns the decision of what happens to the model afterward (saving, logging, etc.) — this separation-of-responsibilities reasoning was worked through explicitly.

**`NUM_EPOCHS` lives in `config.py`**, not hardcoded in `main()` — reasoning: even though it's used only once (in this file), the deciding question was "would any other file ever need to reference this same number," not "how many times does it run." Alex initially reasoned from the wrong angle ("runs once, so hardcode") and self-corrected after the reframe.

**`batch_size=32` is hardcoded directly in `setup()`, not in `config.py`** — opposite conclusion from `NUM_EPOCHS`, and deliberately so. Alex traced through whether `evaluate.py` or `main.py` (pipeline) would ever need to know the training batch size and correctly concluded no — evaluation doesn't need gradient updates or batching in the same sense, and single-song inference isn't batched at all. Unlike `SR`/`N_MELS`, which are true cross-file *consistency requirements* (mismatched values would produce structurally wrong spectrograms), `batch_size` has no such requirement. This is intentionally different from `NUM_EPOCHS`'s placement — both are defensible, the point was reasoning about *why*, not the specific outcome.

### Bugs caught and fixed during this session (for pattern recognition going forward)
1. Conflated dataset and dataloader — wrote `dataloader = OpenMICDataset(...)` more than once, mixing up which object does which job. Root cause: naming a variable correctly doesn't make it that object.
2. Wrote Java/C++-style typed parameters in a function signature (`def train(Dataloader dataloader, Model model, ...)`) — Python signatures don't declare types inline; optional type hints use `name: Type` (colon after, not before).
3. Reached for `self.` to return multiple values from a plain function — `self.` is exclusively a class/instance-attribute mechanism; plain functions return multiple values via tuple (`return a, b`), unpacked positionally by the caller (`x, y = f()`).
4. Left a stray, irrelevant import (`from torch._dynamo.types import CacheEntry`) and `import nn` (not a real standalone package — `nn` is a submodule, correctly `from torch import nn`).
5. Left an unused import (`Dataset` alongside `OpenMICDataset`) — caught and removed.
6. Invented a nonexistent `.sum(min=1.0)` argument trying to solve the div-by-zero edge case — `.sum()` takes no such parameter; the correct denominator is just `label_mask.sum()` with no arguments, and the edge case was already decided to not need guarding.
7. Typo: `total_lost` vs `total_loss` (would have thrown `NameError`).
8. The epoch-loss-tracking bug sequence described above (printing per-batch, then printing stale `loss` after the loop, before landing on accumulate-then-average).

---

## Flagged, Not Yet Resolved (explicitly deferred, time crunch)

- **`CHECKPOINT_PATH = Path("models/classifier.pt")` is a relative path.** Not yet confirmed what directory this resolves to when actually run on Colab, or whether that matches where Alex will look for the file afterward (to download, or feed into `evaluate.py`). Flagged as a "confirm before/during the Colab run" item, not solved.
- **Mel cache accessibility from Colab** — need to confirm the precomputed cache (`data/openmic/mel_cache/`) is actually reachable from the Colab environment (uploaded to Drive, mounted, etc.) before kicking off training.
- **Distribution shift** (training on full OpenMIC mixes vs. inferring on Demucs-separated "other" stem) — flagged repeatedly across sessions (Day 9, Day 11, Day 12) as something to revisit before `evaluate.py`/`classifier.py` design, and definitely before interview prep. Still not revisited. Don't let this slip further.
- **GAP (Global Average Pooling)** — conceptually landed during Day 11's rebuild session but flagged as still a slightly soft spot. No action needed unless it resurfaces naturally.
- **`precompute_mel_cache.py` redundancy** — known, deliberately left as-is (one-time throwaway script, different output contracts make sharing logic with `__getitem__` impractical). Not a bug, a closed decision — don't reopen unless something breaks.

---

## Immediate Next Steps

1. **Run training on Colab (L4 GPU)** using the precomputed mel cache. Confirm the two flagged items above (checkpoint path, cache accessibility) before/during the run.
2. Watch the per-epoch average loss print — confirm it's trending downward across epochs as a basic sanity signal.
3. Once training completes and `classifier.pt` is saved and downloaded/accessible: **build `evaluate.py`** — per-class precision/recall/F1 against the ~5,100 test-partition clips (concept was scoped at a high level back on Day 11: load cached test-partition mel specs → load trained weights → run inference → threshold predictions → compare against `Y_true` respecting `Y_mask` → compute per-class P/R/F1 → report as a table). This has not been scoped in implementation detail yet — will need the same skeleton-first planning process as `train.py`.
4. Before or during `evaluate.py` design: **revisit distribution shift** — this has been deferred three sessions running.

---

## Remaining Build Order (locked in since Day 11)

| Stage | Status |
|---|---|
| `config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py` | ✅ Done |
| `train.py` | ✅ Done (this session) |
| Colab training run | ← **current step** |
| `evaluate.py` | Next |
| `classifier.py` | Not started |
| `main.py` (pipeline wiring) | Not started |
| Full-stack web app | Post-MVP |
| Gemini benchmark comparison | Post-MVP, stretch goal / portfolio talking point |

---

## Process Notes for Next Session

- Continue the file-by-file pattern: state the file's job in one sentence, sketch function/class skeleton only, work through edge cases via guided questions, Claude supplies distilled library references as needed, Alex writes the implementation, Claude reviews after.
- Alex is currently on a stated time crunch — sessions may move faster / skip some of the deeper Socratic exploration that happened in earlier days (e.g. Day 11's full rebuild session). Keep pace reasonable but don't compromise on catching real bugs.
- Continue enforcing: Claude never writes implementation code, even under time pressure — the library-lookup exception (e.g. `DataLoader`'s constructor signature) is for genuine one-off doc lookups, not a general permission to hand over logic or design.
