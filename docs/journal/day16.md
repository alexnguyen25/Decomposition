# Handoff — Decomposition
**After session:** `classifier.py` fully designed, written, and run end-to-end for the first time; real output shows over-prediction on unseen instruments
**Next session focus:** Diagnose why the model over-predicts instruments not present in the song, before treating `classifier.py` as done

---

## Project Overview

**What it is:** An audio intelligence pipeline that takes a music file and outputs 4 separated stems (vocals, drums, bass, other), instrument labels for the "other" stem, BPM, and key — as structured JSON.

**Pipeline:** Preprocessing (validate/normalize) → Demucs source separation → mel spectrogram feature extraction → custom CNN classifier (trained on OpenMIC-2018) → structured output.

**User:** Alex, rising sophomore at Northeastern University, targeting FAANG/top tech co-ops for Spring/Summer 2027. This is the primary portfolio project.

**Claude's role: researcher and tutor only — never write code for Alex.** Socratic method — ask guiding questions, let Alex reason to answers, correct precisely when wrong, use interactive visualizers for spatial/mechanical concepts. Alex owns all architecture decisions, tensor shape reasoning, and writing the actual implementation code. Library/doc lookup is the one narrow exception (used this session to pull `librosa.feature.rms`'s exact signature).

**User preference:** Never push or commit anything without permission.

---

## Current Project State

**Built, verified, and reviewed:**
- `src/config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py`, `train.py`, `evaluate.py` — all from prior sessions
- **`src/classification/classifier.py` ✅ NEW — fully written, reviewed, and run end-to-end this session** (full breakdown below)
- **`src/separation/stem_presence.py` ✅ NEW — `stem_energy` + `is_stem_silent`, written and sanity-checked against real audio this session** (full breakdown below)

**Eval numbers now in hand** (from the stalled run finally completing — it was slow, not hung):
- Macro-F1: 0.650 across 20 classes
- Best performers: cymbals (0.918), drums (0.920), synthesizer (0.906), piano (0.906), guitar (0.875), voice (0.877)
- Worst performers: flute (0.263), accordion (0.272), clarinet (0.293)
- **Key finding this session:** low F1 does not correlate with sample count (ruled out — worst classes had *more* confirmed labels than the best ones). It correlates with **acoustic family confusability** — wind/reed instruments (clarinet, flute, accordion, saxophone) and to a lesser extent brass/strings score worse because they're easily confused *with each other* in a full mix, not because the model can't detect "a wind instrument is present." A single misclassification (true=clarinet, predicted=flute) damages clarinet's recall AND flute's precision simultaneously — this is why weak classes cluster by family rather than appearing as isolated failures.
- **Threshold decision, now locked:** 0.5 stays as the default threshold. Reasoning confirmed correct: the weak classes suffer from a ranking problem (wrong instrument scores highest), not a calibration problem — no threshold, global or per-class, fixes a ranking error. Per-class threshold tuning was explicitly rejected as a next step because it would require a separate validation split never touched by threshold-tuning, which doesn't currently exist; tuning against the test set itself would be a methodological error (same category as test-set leakage).

**Not yet built:**
- `main.py` (top-level pipeline wiring)
- Full-stack web app (post-MVP)
- Gemini benchmark comparison (post-MVP, stretch goal)

---

## `stem_presence.py` — New File This Session

**One-sentence job:** given a stem's waveform and which stem it is, decide whether that stem is basically empty (Demucs found ~nothing there).

**Why a new file, not folded into `classifier.py` or `separation.py`:** `classifier.py` is scoped tightly to the "other" stem and CNN inference — this touches vocals/drums/bass and has nothing to do with the model. It's conceptually closer to `separation.py`'s territory (operates on the stems Demucs produced) but distinct enough, and potentially large enough once calibrated, to warrant its own file.

```python
"""Check whether a Demucs stem is basically empty using framed RMS."""

import librosa
import numpy as np

# TBD — calibrate on real Demucs stems across multiple songs
FRACTION_ABOVE_FLOOR = 0.05
STEM_FLOORS = {
    "vocals": 0.01,
    "drums": 0.01,
    "bass": 0.01,
    "other": 0.01,
}


def stem_energy(y):
    """Return framed RMS values for a waveform."""
    if y.ndim > 1:
        y = y.mean(axis=0)
    return librosa.feature.rms(y=y)[0]


def is_stem_silent(y, stem_name):
    """True if too few frames are above this stem's energy floor."""
    rms = stem_energy(y)
    floor = STEM_FLOORS[stem_name]
    loud_fraction = (rms > floor).mean()
    return loud_fraction < FRACTION_ABOVE_FLOOR
```

### Key design points

**Compute vs. policy split** — `stem_energy` is pure computation (waveform in, RMS-per-frame out), no opinions about what counts as silent. `is_stem_silent` holds the policy (floor lookup, fraction cutoff). This mirrors the same split used in `evaluate.py` (`compute_metrics` vs. `print_report`).

**Per-stem floor dict, not a single global constant** — deliberate, even though all four values are currently identical. Reasoning explicitly worked through: if `bass` sits quieter than `drums` even when genuinely present, a single global threshold risks misclassifying a real bass line as silent. The dict structure supports differentiation later; the *values* are honestly labeled as placeholders ("honest debt"), not invented-to-look-finished numbers.

**Frame-level floor + fraction-above-floor, not whole-file RMS, not max-frame-energy** — both alternatives were considered and correctly rejected:
- Whole-file RMS was rejected because a stem that's silent for half the song and present for the other half would get averaged into a misleadingly low number.
- Max-frame-energy was rejected via a concrete adversarial case: a single brief Demucs bleed/artifact spike in an otherwise silent stem would make max-energy say "present" when it should say "silent." Fraction-of-frames-above-floor is robust to this since one spike is a negligible fraction of total frames.

**Mono collapse via `y.mean(axis=0)`** — confirmed correct against Day 12's already-established fact that `librosa.load(mono=False)` returns `(n_channels, n_samples)`, so channels sit on axis 0.

**Two nested TBD thresholds, not one** — worth naming explicitly since it's easy to lose track of: the per-frame floor (what counts as "loud" for one frame) and the fraction-of-frames cutoff (what fraction of loud frames counts as "stem present") are two separate constants, both currently placeholders, both requiring real calibration data.

### Sanity-checked against real audio this session

Ran `stem_energy`/`is_stem_silent` against a mix of near-empty stems and stems with real content from actual test audio:

| Stem type | Median RMS | Max RMS | `is_stem_silent` @ 0.01 |
|---|---|---|---|
| Near-empty (smoke test vocals/drums/bass) | ~0.0003–0.001 | ~0.001–0.003 | True |
| Present (smoke test "other") | ~0.019 | ~0.020 | False |
| Present (real test stems) | ~0.04–0.14 | ~0.18–0.35 | False |

**Read on this result:** 0.01 sits comfortably between the near-empty cluster (~1e-3) and clearly-present cluster (~0.04+) — not obviously broken. But explicitly flagged as *not* a real calibration: one song's numbers don't establish a general rule. In fact, the real test track's bass stem did **not** show quieter RMS than its drums stem, directly contradicting the "bass sits quieter than drums" acoustic assumption used to justify the per-stem dict structure — a concrete demonstration of why inventing differentiated-looking floor values from one example would have been a mistake.

**Still TBD, blocking full calibration:** need RMS values from multiple real Demucs-separated songs across different genres/mixes (something sparse, something dense, something with a genuinely quiet bass part) before the floor and fraction constants can be considered real rather than placeholder.

---

## `classifier.py` — New File This Session

**One-sentence job:** given a path to a Demucs "other" stem and a loaded model, return a list of detected instruments (name + confidence) for that stem, excluding the four classes Demucs already resolves by construction.

```python
"""Classify instruments in a Demucs "other" stem."""

import json
from pathlib import Path

import numpy as np
import torch

from src.classification.model import Model
from src.config import CHUNK_FRAMES, HOP_LENGTH, OPENMIC_DIR, SR
from src.feature_extraction.feature_extraction import extract_mel_spectrogram

# drop remainders shorter than ~3 seconds
MIN_REMAINDER_FRAMES = int(3 * SR / HOP_LENGTH)

# Demucs already covers these — don't report them from the CNN
DEAD_WEIGHT = {2, 5, 6, 19}  # bass, cymbals, drums, voice


def load_model(checkpoint_path, device):
    """Build Model, load weights, set eval mode."""
    model = Model().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def chunk_mel(mel):
    """Split (128, T) mel into a list of (1, 128, 431) tensors."""
    chunks = []
    t = mel.shape[1]
    start = 0

    while start < t:
        end = start + CHUNK_FRAMES
        piece = mel[:, start:end]
        width = piece.shape[1]

        if width == CHUNK_FRAMES:
            chunk = piece
        elif width >= MIN_REMAINDER_FRAMES:
            padded = np.zeros((mel.shape[0], CHUNK_FRAMES), dtype=mel.dtype)
            padded[:, :width] = piece
            chunk = padded
        else:
            break

        chunks.append(torch.from_numpy(chunk).unsqueeze(0).float())
        start = end

    return chunks


def predict_chunks(chunk_list, model, device):
    """Run all chunks in one batch; return probs shaped (C, 20)."""
    batch = torch.stack(chunk_list).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)
    return probs.cpu()


def aggregate(probs):
    """Max-pool over chunks → song-level probs (20,)."""
    return probs.max(dim=0).values


def format_instruments(song_probs, threshold=0.5):
    """Threshold, drop dead-weight classes, attach names."""
    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    index_to_name = {index: name for name, index in class_map.items()}

    instruments = []
    for i, prob in enumerate(song_probs.tolist()):
        if i in DEAD_WEIGHT:
            continue
        if prob >= threshold:
            instruments.append({"name": index_to_name[i], "confidence": prob})
    return instruments


def classify(other_path, model, device):
    """other stem path → list of {name, confidence} for non-Demucs instruments."""
    mel = extract_mel_spectrogram(Path(other_path))
    chunks = chunk_mel(mel)
    if not chunks:
        return []
    probs = predict_chunks(chunks, model, device)
    song_probs = aggregate(probs)
    return format_instruments(song_probs)
```

### Key design points

**Six functions, matching the skeleton locked earlier in the session** — `load_model`, `chunk_mel`, `predict_chunks`, `aggregate`, `format_instruments`, `classify` (thin orchestrator). Each has a single, narrow job, same pattern as `train.py`/`evaluate.py`.

**`chunk_mel`'s stacking boundary, explicitly reasoned through before writing:** stacking chunks into a batch tensor is a "prepare for the forward pass" operation, not a "cut a spectrogram" operation — so it belongs in `predict_chunks`, not `chunk_mel`. `chunk_mel` returns a plain list of individual `(1, 128, 431)` tensors; `predict_chunks` does the `torch.stack`.

**Aggregation order (sigmoid-then-max vs. max-then-sigmoid) — resolved as a non-issue, correctly:** sigmoid is monotonic, so `max(sigmoid(x))` and `sigmoid(max(x))` pick out the identical chunk and produce identical results. This was explicitly worked through rather than assumed — it's a convention choice, not a correctness one. Code applies sigmoid first (matches `evaluate.py`'s convention).

**Filtering (`DEAD_WEIGHT`) lives in `classifier.py`, not alongside `is_stem_silent` in `stem_presence.py`, and this split was explicitly justified, not arbitrary:** dead-weight filtering is about the CNN's fixed 20-class output contract — inherent to the model itself, so it's natural for `classifier.py` to own it. Silence detection is a different input (raw waveform, not "other" stem mel) and a different kind of judgment (energy threshold, no model involved) — a genuinely separate concern with a separate natural home.

**`chunk_mel`'s numpy slicing behavior, traced by hand, not assumed:** confirmed that slicing past the end of a numpy array (e.g. `mel[:, 431:862]` when `t=500`) does not error — it silently clips to `mel[:, 431:500]`. The `width` check after slicing correctly handles this because it checks the *actual* resulting width, not the requested `end` index. Flagged as a generally important fact to remember: silent-wrong behavior (no error, just a smaller-than-expected array) is more dangerous than loud-wrong, since nothing signals it happened.

**Empty-chunks guard in `classify` — traced through and correctly classified as defensive, not dead, code:** through the real pipeline (validation already enforces ≥10s minimum), `chunk_mel` will always return at least one chunk, making the guard unreachable in normal operation — similar to Day 12's dead 3D branch in `readChannelCount`. But unlike that case, this guard is **not** dead code: `classify()` could plausibly be called directly by something that skips validation, and `torch.stack([])` would crash without the guard. Correctly identified as "crash insurance for a standalone-API edge case," a different verdict than the truly-dead 3D branch, with the reasoning that produces the distinction made explicit.

**`class-map.json` re-opened on every `format_instruments` call — identified as a real but low-priority inefficiency, not fixed yet, with an explicit criterion for when to fix it:** cost is negligible next to mel extraction and CNN inference for occasional single-song calls. Correctly identified the actual trigger condition for revisiting this: if profiling shows `classify()` is genuinely hot (e.g., batch-processing many songs) or if the JSON file itself lives on slow storage (e.g., Drive) during batched calls. Until then, left as-is by design, not by oversight.

---

## First End-to-End Run — Real Result, Real Problem Surfaced

Ran `classify()` against a real separated "other" stem (`data/test_audio/test/other.wav`) for the first time this session. Completed in ~2.5s on CPU (confirms the earlier reasoning that inference-only workloads don't need GPU/Colab).

**Output:** 13 instruments above the 0.5 threshold (dead-weight classes already filtered):

| Instrument | Confidence |
|---|---|
| synthesizer | 0.947 |
| guitar | 0.821 |
| violin | 0.695 |
| trumpet | 0.681 |
| saxophone | 0.667 |
| mandolin | 0.662 |
| banjo | 0.645 |
| piano | 0.641 |
| mallet_percussion | 0.637 |
| cello | 0.591 |
| trombone | 0.547 |
| ukulele | 0.537 |
| flute | 0.510 |

**Ground truth, obtained by looking up the actual song's known instrumentation** (not relying on ear alone, which was explicitly flagged this session as weaker evidence — bleed and imperfect separation make "what I hear" an unreliable check): vocals, lead guitar, rhythm guitar, bass, drums, piano/synth. Excluding the three dead-weight classes already stripped from the report (bass, drums, voice), the real non-Demucs-covered instrumentation is: **guitar, piano/synth** — nothing else.

**Comparison:**
- **Correct hits:** synthesizer (0.947), guitar (0.821), piano (0.641) — all genuinely present, reasonably high confidence.
- **False positives, no basis in the actual song:** violin, trumpet, saxophone, mandolin, banjo, mallet_percussion, cello, trombone, ukulele, flute — 10 of 13 reported instruments.

**This is not a new failure mode — it's the eval-table finding (family confusability among wind/brass/string instruments) now visible in a real deployment scenario rather than as an aggregate test-set statistic.** Nearly every false positive falls into the same acoustic family cluster already identified as weak in per-class eval metrics (violin 0.827 F1 — actually one of the stronger classes, worth noting as a partial exception; trumpet 0.613, trombone 0.523, saxophone 0.706, banjo/mandolin/cello/ukulele/flute all middling-to-poor). The model correctly identified what's genuinely there but also fired confidently on roughly a dozen instruments that aren't present at all — this looks less like calibration noise and more like a real over-prediction problem on this input, not yet explained.

**Not yet resolved — flagged as the explicit next-session starting point:** why does a distorted rock guitar's spectral content apparently trigger confident (0.5–0.7 range) false positives across nearly the entire wind/brass/string cluster simultaneously? This was raised as an open question at the end of the session (does guitar's harmonic content plausibly resemble, e.g., cello or trombone's spectral shape to a CNN operating near its confusion boundary?) but not worked through yet.

---

## Immediate Next Steps

1. **Diagnose the over-prediction problem** surfaced by this session's real test run. Concrete angles worth pursuing, not yet attempted:
   - Look at raw pre-threshold probabilities across *all 20* classes for this song (not just the >0.5 ones) — are the false positives clustered just above 0.5, suggesting borderline confusion, or confidently high, suggesting something more structurally wrong?
   - Test against 2–3 more real songs with known instrumentation to see if this is a one-off or a pattern — one bad result doesn't yet establish "the model over-predicts," same evidentiary standard already applied earlier this session to the eval table and the RMS calibration.
   - Revisit whether this connects to the already-accepted distribution shift (training on full OpenMIC mixes vs. inferring on a Demucs "other" stem with different spectral characteristics than anything seen in training) — this may be the distribution shift manifesting concretely for the first time, rather than a separate new bug.
2. **Calibrate `stem_presence.py`'s floor/fraction constants** against multiple real Demucs-separated songs — currently placeholder values, sanity-checked against only one track.
3. Once the over-prediction issue is understood (fixed, mitigated, or consciously accepted as a documented limitation): move to `main.py`.

---

## Remaining Build Order

| Stage | Status |
|---|---|
| `config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py`, `train.py` | ✅ Done |
| Colab training run | ✅ Done |
| `evaluate.py` (code + real run) | ✅ Done — Macro-F1 0.650, family-confusability finding documented |
| `stem_presence.py` | ✅ Written, placeholder calibration only |
| `classifier.py` (code + first real run) | ✅ Written — **real run surfaced an unresolved over-prediction problem** |
| Diagnose over-prediction issue | ← **current step, next session** |
| `main.py` (pipeline wiring) | Not started |
| Full-stack web app | Post-MVP |
| Gemini benchmark comparison | Post-MVP, stretch goal / portfolio talking point |

---

## Process Notes for Next Session

- Continue the file-by-file, function-by-function pattern: one sentence of purpose, skeleton first, edge cases via guided questions, library lookups as the one exception, Alex writes, Claude reviews after.
- This session repeatedly demonstrated a good instinct worth reinforcing: testing a plausible-sounding theory against actual data before accepting it (the "frequent classes score better" theory was proposed, then correctly discarded once checked against `n_confirmed` counts, which ran the opposite direction). Keep applying this same standard to the over-prediction diagnosis next session — don't accept the first plausible explanation without checking it against real numbers.
- Alex correctly distinguished, twice this session, between "dead code" and "defensive code for an edge case that's unreachable through the normal pipeline but reachable through direct/standalone calls" — worth continuing to ask this distinction explicitly whenever a guard clause or edge-case branch shows up in review.
- Watch for over-trusting ear/intuition over ground truth — flagged explicitly this session when Alex's own listening impression of the test song was checked against (and partly contradicted by) the song's actual documented instrumentation.
