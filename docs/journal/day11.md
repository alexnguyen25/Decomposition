# Dev Journal — Day 11
**Date:** July 1, 2026
**Project:** Decomposition

---

## What We've Done Today

No code written today — this was a full comprehension rebuild session across two days (continued from a session yesterday), following the two-week gap plus a stretch of backtracking during implementation. Goal was to fix a root cause: build breadth + depth understanding of the entire pipeline before touching any new files, rather than learning stage-by-stage and hitting surprises mid-build.

Also introduced a new scope addition: an **eval harness** (`evaluate.py`) as a formal pipeline stage.

---

## Session Format

Pure Socratic review — no new concepts introduced without first being pulled out of Alex via questioning. Alex explained the pipeline from memory multiple times across the session, with corrections applied each pass. Confidence self-rated as "better but still shaky" after the first pass, "solid" by the end of the second.

---

## Conceptual Ground Covered

**Sample rate split (corrected for the 4th time — now locked in via first-principles derivation, not memorization)**
- 44,100Hz belongs to **Demucs** — hardcoded requirement of the pretrained model, needed for full-spectrum source separation
- 22,050Hz belongs to the **classifier path** — matches OpenMIC-2018's native rate, halves compute, instrument identity survives the ~11kHz cutoff
- Derived the mechanism from scratch: feeding Demucs a lower sample rate makes it hear audio at half speed / an octave low, since Demucs assumes 44,100 numbers = 1 second regardless of the file's actual rate
- Confirmed why you can't halve again to ~11kHz: OpenMIC is fixed at 22,050Hz, and train/inference sample rate must always match — tradeoffs are only valid if consistent across both

**Stereo requirement**
- Demucs uses phase differences between L/R channels for separation; mono removes that spatial information

**Mel spectrogram, rebuilt from scratch**
- Confirmed it's a 2D grid (frequency × time), each cell holding **power** (energy) as its value — same conceptual object as a grayscale image's pixel grid
- `power_to_db`: does NOT change axes, only rescales the power values from a huge dynamic range into a compressed, learnable range. Not related to human hearing — that's the mel *scale's* job, not the dB step's
- Derived `(1, 128, 431)` from first principles: 128 = `n_mels` (a chosen hyperparameter), 431 = time frames = `(10s × 22,050) / 512 hop_length` ≈ 431, 1 = channel dimension (grayscale-equivalent)

**Conv2D, worked through with literal arithmetic**
- Manually computed dot products between a 3×3 filter and a 3×3 patch to prove the mechanism (uniform patch → filter with alternating signs → output of 0; patterned patch → high output)
- Key correction: filters detect **shape/pattern match**, not "how big the numbers are" — a flat region scores 0 even with large uniform values
- Horizontal band pattern = sustained pitched note (violin, held piano chord); vertical band = percussive transient (drum hit)

**ReLU**
- Zeroes negative dot products (which indicate "opposite of this filter's pattern"); positive values pass through unchanged. Other filters exist to detect the inverse pattern directly, so this filter doesn't need to track negatives

**MaxPool**
- Takes the max value in each small local block (e.g. 2×2) of an already-computed feature map — does NOT recompute dot products
- Rationale for max over average at this stage: preserves the peak detection signal in a small local region; also gives translation invariance (a note landing in frame 3 vs frame 4 of the same block gives the same output)

**Two-layer stacking — "patterns of patterns" (the main sticking point this session, resolved via a custom visual diagram)**
- Layer 1 filters detect simple shapes on the raw spectrogram (bands, spikes, harmonic stacks)
- Layer 2 filters run on Layer 1's *feature maps*, not the spectrogram — detecting combinations/co-occurrences of Layer 1 patterns (e.g. harmonic stack + pluck onset + sustained band, together, in the same region = guitar-like signature)
- Neither layer "knows" what a guitar is — that mapping only happens in the final Linear layer

**GAP (Global Average Pooling) — still the softest spot conceptually**
- Averages every value in each `32×107`-ish feature map down to one number per channel → 64 numbers total, one per learned feature
- Average (not max) chosen here specifically for robustness: prevents one brief artifact/spike from dominating the whole-clip score; MaxPool (local, repeated, mid-network) and GAP (global, once, final) are answering different-scale questions — "did this fire nearby" vs. "is this present overall in the whole clip" — not opposites, sequential zoom-outs
- Flagged explicitly as needing more exposure before it fully clicks — revisit if it resurfaces

**Linear(64→20) and Sigmoid — closed the full loop this session**
- Rebuilt from scratch that this is 20 independent weighted sums (each output/instrument has its own full set of 64 weights, learned in training) — NOT sorting or bucketing
- 64 × 20 ≈ 1,280 weights total, vs. ~4M if flatten had been used instead of GAP
- Sigmoid squashes each raw score independently to 0–1 — a probability, not a rounding operation. Thresholding at 0.5 is a separate, subsequent step
- Reconfirmed sigmoid vs. softmax: softmax forces outputs to sum to 1 (mutually exclusive, "pick one"); sigmoid treats each instrument independently — required here since multiple instruments can co-occur in one stem

**Distribution shift** — stashed for later by Alex's request, not covered in depth this session. Flag to revisit: training happens on full OpenMIC mixes, inference happens on the Demucs-separated "other" stem (missing drums/bass/vocals, contains separation artifacts).

**Silent-stem edge case (new, resolved this session)**
- Confirmed Demucs always outputs all 4 stems regardless of content — if a song has no drums, the drums stem is just near-silent audio, not an error condition. Nothing breaks; this is a natural edge case for the eval harness to characterize, not a pipeline failure mode.

**Eval harness concept (new scope, planned at a high level for the first time)**
- Established why raw accuracy is meaningless here: heavy class imbalance across 20 instruments means "always predict absent" scores deceptively high
- Introduced precision, recall, F1 — computed **per instrument class**, not in aggregate — to reveal which instruments the model handles well vs. poorly
- Sketched the eval harness pipeline: load cached test-partition mel spectrograms (already exist from the precompute cache — no new extraction needed) → load trained model weights → run inference on all test clips → threshold predictions → compare against `Y_true` (respecting `Y_mask`) → compute per-class precision/recall/F1 → report as a table

**Music AI API landscape (researched live, tangential/portfolio note)**
- Confirmed OpenAI's audio API has no instrument-classification capability — it's built for transcription/TTS/realtime voice, not music analysis
- Identified that dedicated music AI APIs (e.g. Cyanite) do this exact task commercially, converting tracks to spectrograms and running multi-label classifiers per time segment — validates the architecture choice and opens a possible future benchmark: compare Decomposition's classifier output against a production MIR API. Noted as a portfolio idea, not scheduled work.

---

## Current Project State (unchanged code-wise from Day 10)

**Built and verified:**
- `src/config.py`
- `src/preprocessing/validation.py`
- `src/preprocessing/processing.py`
- `src/separation/separation.py`
- `src/feature_extraction/feature_extraction.py`
- `src/classification/dataset.py`

**Not yet built:**
- Cache script fix (the `OpenMICDataset`-instantiated-but-bypassed redundancy is still an open, unresolved design question — intentionally deferred, not forgotten)
- `model.py`
- `train.py`
- `evaluate.py` (new)
- `classifier.py`
- `main.py`

---

## Full Remaining Build Order (locked in)

1. Resolve cache script redundancy → run precompute on Colab
2. `model.py` — CNN architecture
3. `train.py` — masked BCEWithLogitsLoss training loop, run on Colab L4
4. `evaluate.py` — per-class precision/recall/F1 against the ~5,100 test clips
5. `classifier.py` — chunked inference on real songs, cross-chunk max-pool aggregation
6. `main.py` — full pipeline wiring, JSON output

---

## Process Change Agreed for Upcoming Sessions

Alex found independently researching PyTorch function signatures (e.g. for `dataset.py`) low-value relative to time spent — it didn't noticeably help retention. Going forward:

- **Claude will look up and distill exact library function signatures/behavior** (e.g. `nn.Conv2d`, `nn.AdaptiveAvgPool2d`, `nn.Linear`) rather than sending Alex to read raw docs
- **Alex still owns**: deciding what layers/functions are needed and in what order, reasoning through tensor shapes at each step, writing the actual implementation, and making design-tradeoff calls
- This is a targeted change to *one* sub-step of the workflow (doc lookup), not a change to the core rule that Claude doesn't write project code

---

## Next Session Plan

1. **Code review pass** — walk through each existing file (`validation.py`, `processing.py`, `separation.py`, `feature_extraction.py`, `dataset.py`) together. Alex explains the logic; Claude checks understanding and fills gaps. Not a fresh read — a verification pass before building on top of this code.
2. **Resolve the cache script redundancy** — don't skip this because it feels small; getting it wrong now means debugging a training pipeline built on a hacked-around fix later.
3. **Low-level planning for each remaining file**, using this sequence per file:
   - State the file's job in one sentence (input → output)
   - Sketch function/class signatures (skeleton only, no implementation)
   - Work through edge cases and design decisions via guided questions
   - Claude supplies distilled library function references as needed
   - Alex writes the implementation; Claude reviews after
4. Suggested checkpoint: do a confidence/shape-tracing check specifically after `model.py`, since silent shape mistakes there are easy to make and hard to catch until training breaks.

## Topics to Revisit (flagged, not urgent)

- GAP — conceptually landed but still softer than the rest; give it more exposure rather than forcing it
- Distribution shift — stashed by Alex's request; revisit before evaluate.py/classifier.py design, and definitely before any interview prep
