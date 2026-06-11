Dev Journal — Day 9
**Date:** June 10, 2026
**Project:** Decomposition

---

## What We've Done Today

- Full plan review of Days 9–13 with Opus 4.8 — caught several issues before implementation
- Resolved all open architecture decisions from Day 8
- Inspected the OpenMIC-2018 dataset for the first time — extracted the `.tgz` and explored the npz structure
- Designed the Dataset class (not yet implemented)

---

## Architecture Decisions Finalized

**Sample rate split**
- Demucs and preprocessing stay at 44,100 Hz — required by the pretrained model
- Classifier path (mel computation, OpenMIC loader) runs at 22,050 Hz — halves compute, instrument identity lives below 11 kHz anyway
- At 22,050 Hz a 10-second clip is exactly 431 frames — the original architecture diagram is now self-consistent

**Single source of truth for mel**
- `extract_mel_spectrogram` is the one place the mel recipe lives — both the OpenMIC loader and inference call it
- Added `power_to_db` step to `extract_mel_spectrogram` — log scaling improves CNN training, and doing it inside the shared function means both paths inherit it automatically
- Load stems at `sr=22050` instead of `sr=None`

**Global Average Pooling replaces flatten**
- After the second conv+pool the feature grid is `(64, 32, 107)`
- GAP averages each channel's grid to one number → 64-element vector → `Linear(64→20)` ≈ 1,280 weights vs 4.4M previously
- Average pooling within each chunk (smooths Demucs separation artifacts), max pooling across chunks for song-level output (catches instruments present in only part of a song)
- `nn.AdaptiveAvgPool2d` is the PyTorch function to use

**Chunking and minimum duration**
- 10-second minimum validation replaces the 5-second minimum — guarantees every valid input produces at least one full chunk
- Zero-pad the trailing remainder; drop remainders under ~3–4 seconds to avoid mostly-silence chunks
- With GAP in place, exact chunk width is no longer hardcoded — shape errors from slightly-off chunks go away

**Shared config**
- All mel parameters (`SR = 22050`, `N_MELS = 128`, `N_FFT = 2048`, `HOP_LENGTH = 512`, `CHUNK_FRAMES = 431`) live in one config file — everything imports from there

**Distribution shift (accepted for v1)**
- Training on full OpenMIC mixes, inferring on the Demucs "other" stem — the stem is a reduced, artifact-affected mix the model never saw in training
- Columns 2, 5, 6, 19 (`bass`, `cymbals`, `drums`, `voice`) are dead weight at inference — Demucs strips those out
- Accepted tradeoff for MVP; can articulate it clearly in an interview

---

## OpenMIC-2018 Dataset Inspection

**Structure confirmed:**
```
openmic-2018/
├── audio/000/000046_3840.ogg ...   ← raw audio, path: audio/{key[:3]}/{key}.ogg
├── openmic-2018.npz                ← arrays: X, Y_true, Y_mask, sample_key
├── class-map.json                  ← instrument → column index (0–19)
├── partitions/
│   ├── train01.txt                 ← ~14,900 sample keys
│   └── test01.txt                  ← ~5,100 sample keys
```

**npz arrays:**
- `X` — VGGish embeddings, ignore entirely
- `Y_true` — shape `(20000, 20)`, confidence scores 0.0–1.0, threshold at 0.5 for binary labels
- `Y_mask` — shape `(20000, 20)`, booleans — `True` means label is confirmed and contributes to loss
- `sample_key` — shape `(20000,)`, string array, requires `allow_pickle=True`

**Label format:** not binary — crowd-sourced confidence scores. `0.5` means "no confident verdict." Confirmed labels sit at `0.0`, `1.0`, or intermediate values. Threshold at `>= 0.5` → present.

**Class map:** 20 instruments in alphabetical order — accordion (0) through voice (19).

**Partition format:** plain text, one sample key per line, no headers.

---

## Key Concepts Learned

**`np.isin` for filtering**
Returns a boolean array of the same shape as the input where `True` means the element exists in the test set. Use it to filter npz rows to one partition: `mask = np.isin(sample_key, partition_keys)` → `Y_true[mask]`, `Y_mask[mask]`, `sample_key[mask]`. No explicit loop needed.

**Dataset ordering doesn't matter**
`__getitem__` only needs internal consistency — index `i`'s spectrogram, labels, and mask all come from the same clip. A DataLoader shuffles every epoch anyway. The order rows are stored is irrelevant as long as the three arrays are sliced with the same mask.

**Mel parameters produce 431 frames at 22050**
`t = 1 + floor(n_samples / hop_length)` with `center=True`. At 22050 × 10s = 220,500 samples and hop 512: `1 + floor(220500/512) = 431`. Cursor's written formula was wrong (it included `n_fft` in the numerator) even though its verified Python output was correct — always trust the tool's verified output over its derivations.

---

## Struggles

- The "two pooling locations" concept was initially confusing — GAP inside the network vs. max-pool across chunks are different operations in different places serving different purposes
- Took a moment to see why Dataset ordering doesn't matter — instinct was that wrong order = wrong clip, but internal consistency is the only requirement

---

## Action Items for Next Session

- [ ] Create shared config file with mel parameters and SR
- [ ] Update `feature_extraction.py` — `sr=22050`, add `power_to_db`
- [ ] Write the PyTorch Dataset class
- [ ] Implement train/test split using official partition files
- [ ] Write the mel precompute cache script
- [ ] Verify shapes: `(1, 128, 431)`, `(20,)`, `(20,)`

---

## Remaining Roadmap

| Day | Stage |
|-----|-------|
| 9 | OpenMIC data loader + weak label handling ← current |
| 10 | CNN implementation + training loop |
| 11 | Training run + evaluation |
| 12 | JSON output + full pipeline integration in `main.py` |
| 13 | Buffer — bugs, cleanup, README, MVP done |