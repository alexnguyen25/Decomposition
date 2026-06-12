# Dev Journal — Day 10
**Date:** June 11, 2026
**Project:** Decomposition

---

## What We've Done Today

- Completed goals 1 and 2: created `src/config.py` with shared mel parameters, updated `feature_extraction.py` to use `sr=22050` and `power_to_db`
- Researched and understood the PyTorch `Dataset` class contract — `__init__`, `__len__`, `__getitem__`
- Implemented `src/classification/dataset.py` — full `OpenMICDataset` class
- Verified shapes: `(1, 128, 431)`, `(20,)`, `(20,)` all confirmed correct

---

## What We Built

**`src/classification/dataset.py` — `OpenMICDataset` class:**
- `__init__` — loads npz, reads partition txt into a set, filters all three arrays with `np.isin`, stores filtered arrays as instance variables
- `__len__` — returns number of clips in the partition
- `__getitem__` — constructs audio path from key, calls `extract_mel_spectrogram`, adds channel dim with `unsqueeze(0)`, thresholds `Y_true` at >= 0.5 for binary labels, returns spectrogram + labels + mask as tensors

---

## Key Concepts Learned

**PyTorch Dataset contract** — a Dataset class inherits from `torch.utils.data.Dataset` and must implement three methods: `__init__`, `__len__`, and `__getitem__`. The DataLoader wraps it, shuffles indices, and calls `__getitem__` repeatedly to build batches.

**`allow_pickle=True`** — required when loading npz files containing Python objects like strings. numpy refuses to deserialize pickled data without this flag as a security guardrail.

**`np.isin` and boolean indexing** — `np.isin(sample_key, partition_keys)` returns a boolean array the same length as `sample_key`. Applying that mask to numpy arrays keeps only the rows where the mask is True. Applying the same mask to all three arrays guarantees internal consistency — index `i` always refers to the same clip across all arrays.

**`labels` vs `label_mask`** — `labels` is whether an instrument is present (Y_true thresholded at >= 0.5). `label_mask` is whether that label is trustworthy (Y_mask). During training, loss gets multiplied by the mask — unconfirmed labels contribute zero.

**`unsqueeze(0)`** — adds a channel dimension at position 0. Turns `(128, t)` into `(1, 128, t)` — the shape a CNN expects.

**Single source of truth for mel** — `__getitem__` calls `extract_mel_spectrogram` rather than rewriting mel logic. Both the dataset loader and inference path use the same function, so mel parameters are consistent everywhere.

---

## Struggles

- Understanding why `Dataset` needed to be a class at all — clicked once the "contract with DataLoader" framing landed
- Distinguishing `labels` from `label_mask` — both are 20-element tensors but they answer completely different questions

---

## Action Items for Next Session

- [ ] Write the mel precompute cache script
- [ ] Implement the CNN `nn.Module` class
- [ ] Write the training loop with `BCEWithLogitsLoss` and masked loss