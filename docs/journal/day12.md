# Dev Journal — Day 12
**Date:** July 5, 2026
**Project:** Decomposition

-- 

## Random Spiel

Okay I'm finally back and handwriting this one for today. Life got a little busy, but I spent the past few days looking over the project and really understanding everything again. I even planned out some more things after getting this mvp done. The end state of this project should be a full stack website, uploading a song, getting the instruments detected in it, bpm, key, basic info like that from demucs and our classifier. Then I want to also use Gemini's api to create a description of the song, and compare it with what we have detected vs what it detects. I'm going to upload random notes I've taken over the past few days while scoping things out and relearning the project

---

## Project Recap (re-derived from scratch)

**High level:** take an audio clip, figure out what instruments are in it. Requires preprocess → separate → extract features → classify.

**Preprocess**
- Enforce >10 sec and <10 min. 10 sec minimum because that's the minimum length needed for a full mel spectrogram clip, and OpenMIC (our training data) is 10-second clips.
- Normalize to 2 channels — Demucs uses phase difference between left/right to help separate stems.
- Normalize to 44,100 Hz — the sample rate Demucs was trained on.

**Separate**
- Demucs splits into 4 stems: drums, voice, bass, other.
- "Other" is where the un-labeled instruments live — that's what our classifier figures out.

**Feature extraction**
- Take the "other" stem, build a mel spectrogram.
- 10 sec clip, 22,050 Hz (halves runtime vs. 44.1kHz; instrument identity mostly lives below ~11kHz anyway).
- Power-to-dB scaling so values are more meaningful to the model and training is more stable.

**Classify**
- Train on OpenMIC: 20k 10-second clips with instrument labels. Some labels are unconfirmed, hence `Y_mask`.
- Precompute all 20k mel spectrograms ahead of training.
- Train on Google Colab.
- Model is a CNN — designed to find patterns in 2D images (a mel spectrogram counts as one).
- Conv2d slides a kernel over the image; each position gets a dot product, higher = stronger pattern match.
- ReLU zeroes out negative dot products (which indicate "opposite of the pattern") — this is fine because other filters exist specifically to detect the inverse pattern directly.
- MaxPool keeps the highest dot product in each block, shrinking the map so the next Conv2d layer can detect patterns-of-patterns.
- After both conv blocks, GAP brings us down to 64 values (one per learned feature).
- A dense layer holds a separate set of weights per instrument, multiplies them against those 64 values, giving a raw score per instrument.
- Sigmoid turns each raw score into a 0–1 confidence value; thresholding is the separate step that turns that into an actual yes/no.

**Eval**
- Get mel specs for the test set → run trained model → save predictions → compare against ground truth → compute per-class metrics → report.

**Main**
- Strings the whole pipeline together. For real songs, `classifier.py` will split audio into 10-second mel spec chunks before running inference.

---

## Questions I Worked Through

**Why do we need a minimum 10-second clip — why not pad a 3-second clip with blank space instead?**
Technically possible, but a 3-second clip padded to 10 seconds would be ~70% silence. We want each clip to carry enough real signal to make a meaningful prediction, not mostly nothing.

**Why does it matter that we match 44.1kHz / 2 channels — just because that's what Demucs was trained on?**
I understood *that* we need to match, but not fully *why*. The answer: feeding a model data in a different format than it was trained on doesn't degrade gracefully — the model's learned patterns don't map cleanly onto structurally different input. It's like showing someone a photo negative and asking them to recognize faces — the underlying structure is different enough that pattern-matching breaks down, even though a human (or model) might expect it to "sort of still work."

**Idea:** what if we ran the audio through something like the Claude API alongside our own pipeline, and compared its interpretation of the song against our own detected instruments — both as a sanity check and as a research angle? (Related to the Gemini comparison idea above — may consolidate these.)

**Why 22,050 Hz for the classifier path and not something lower, like 11k?**
Because OpenMIC's clips are natively 22,050Hz — training data and inference input have to match sample rates, or the mel spectrograms look structurally different from what the model learned on. You don't get to freely pick a rate; you're locked to whatever the training data used.

---

## File-by-File Understanding Check

**validation.py**
- `checkLength` — guards against clips under 10 sec or over 10 min.
- `checkFileFormat` — hardcoded list of non-audio extensions; uses `Path` to pull the extension and rejects if it matches.
- `validAudio` — runs both checks, then loads the file with librosa to get the array + sample rate (this load is also what surfaces corruption, since a corrupted file throws here).

**processing.py**
- `process_audio` — loads the file, normalizes sample rate and channels, builds the output path (`data/processed/<name>_processed<ext>`), writes it out with `soundfile`.
- `normalizeSampleRate` — resamples to 44,100 Hz via `librosa.resample` if needed, otherwise passes through unchanged.
- `normalizeChannels` — calls `readChannelCount`; if channels > 2, slices down to the first 2 via `y[:2, :]`.
- `readChannelCount` — infers channel count from array dimensions: 1D → mono, 2D → `y.shape[0]`. (Confirmed with Claude: since this pipeline only ever processes one file at a time, `librosa.load` can never actually return a 3D array — the 3D "batch" branch in this function is dead code, not a live bug. `y[:2, :]` is correct as-is for the real 2D case.)

**separation.py**
- `separate` — runs Demucs on the file path, returns a dict of stem name → path.
- `_run_demucs` — builds the Demucs CLI command, runs via `subprocess.run(check=True, capture_output=True)`. `check=True` converts bad exit codes into `CalledProcessError`; `capture_output=True` keeps error output out of the console. Raises `FileNotFoundError` if Demucs itself can't be found.
- `_collect_stems` — builds the dict of stem paths that Demucs produces.

**feature_extraction.py**
- `extract_mel_spectrogram` — pulls `sr`, `n_mels`, `n_fft`, `hop_length` from config. `n_mels=128` is the frequency resolution (CNN input height). `hop_length=512` is the step size in samples between frames — combined with `sr=22050` and a 10-second clip, this is what produces 431 output frames. Applies `power_to_db` so the CNN trains more stably on compressed, meaningful values instead of raw power's huge dynamic range.
- `extract_bpm` — runs librosa's beat tracker on the drums stem specifically, since it gives the clearest onset signal.
- `extract_key` — uses Essentia's `KeyExtractor` on the loaded audio.
- `extract_features` — bundles key, BPM, and mel spectrogram into one dict.

**dataset.py**
- `__init__` — loads the OpenMIC `.npz` (`Y_true`, `Y_mask`, `sample_key`), reads the partition file, filters all three arrays using the *same* boolean mask — critical so that index `i` always refers to the same clip across `Y_true`, `Y_mask`, and `sample_key` simultaneously. Filtering each array independently would risk them drifting out of sync.
- `__len__` — number of clips in the partition.
- `__getitem__` — loads a cached mel spectrogram if one exists, otherwise computes it fresh via `extract_mel_spectrogram`; converts to a tensor with a channel dimension (`unsqueeze(0)`); thresholds `Y_true` at ≥0.5 into binary labels; returns `(spec, labels, label_mask)`.

**precompute_mel_cache.py**
- Loops through both train/test partitions, generates a mel spectrogram for every clip, saves it to the cache path.
- **Known redundancy, deliberately left alone:** this script constructs a full `OpenMICDataset` just to read `.sample_key` off of it — paying the cost of loading and filtering `Y_true`/`Y_mask` even though neither is ever used. It also reimplements its own cache-check-and-extract logic rather than calling `__getitem__`, but that duplication turned out to be legitimate: `__getitem__` returns a tensor shaped for model consumption, while the cache needs the raw pre-conversion NumPy array — the two have genuinely different output contracts, so sharing the logic isn't actually a clean option. Decided this isn't worth fixing since it's a one-time throwaway script.

---

## model.py — Planned and Implemented

**Layers:**
- `conv1`: `Conv2d(in=1, out=32, kernel_size=3, padding=1)` — own weights, learnable
- `conv2`: `Conv2d(in=32, out=64, kernel_size=3, padding=1)` — own weights, learnable
- `pool`: `MaxPool2d(kernel_size=2)` — reused, no learnable weights
- `relu`: `ReLU()` — reused, no learnable weights
- `gap`: `AdaptiveAvgPool2d(output_size=(1,1))` — reused, no learnable weights
- `fc`: `Linear(64, 20)` — own weights, learnable (1,280 weights + 20 biases, fully independent per instrument)

**Forward pass:** conv1 → relu → pool → conv2 → relu → pool → gap → `squeeze(dim=(2,3))` → fc → return raw logits (no sigmoid — that's `BCEWithLogitsLoss`'s job during training, and `classifier.py`'s job at real inference time).

**Shape trace (B = batch size):**
```
input                (B, 1, 128, 431)
conv1 → relu → pool  (B, 32, 64, 215)
conv2 → relu → pool  (B, 64, 32, 107)
gap                  (B, 64, 1, 1)
squeeze(dim=(2,3))   (B, 64)
fc                   (B, 20)
```

**Why `squeeze(dim=(2,3))` and not plain `.squeeze()`:** plain squeeze removes *every* size-1 dimension it finds — including the batch dimension, if batch size ever happens to be 1 (e.g. during single-song inference later). Specifying `dim=(2,3)` targets only the two leftover spatial dims from GAP, leaving batch size untouched regardless of its value.

**Bugs I caught and fixed while implementing:**
1. Forgot to inherit from `nn.Module` (`class Model():` → `class Model(nn.Module):`) — needed for `.parameters()`, `.state_dict()`, etc. to exist at all.
2. Missing `super().__init__()` as the first line of `__init__` — required for `nn.Module`'s internal layer-tracking setup to run before attaching any `self.` layers.
3. Layer calls weren't reassigned (`self.conv1(copy)` instead of `copy = self.conv1(copy)`) — calling a layer computes and returns a new tensor, it doesn't mutate in place.
4. Missing the second `self.pool(copy)` call after the second ReLU — caught by tracing shapes against the locked architecture step by step.

---

## Current Project State

**Built and verified:**
- `src/config.py`
- `src/preprocessing/validation.py`
- `src/preprocessing/processing.py`
- `src/separation/separation.py`
- `src/feature_extraction/feature_extraction.py`
- `src/classification/dataset.py`
- `src/classification/model.py` ✅ NEW — CNN architecture, fully implemented and reviewed

**Not yet built:**
- `precompute_mel_cache.py` fix — decision made: leaving as-is (see above)
- `train.py` ← next up
- `evaluate.py`
- `classifier.py`
- `main.py`

---

## Action Items for Next Session

- [ ] Quick sanity check on `model.py`: feed a random `(B, 1, 128, 431)` tensor through and confirm output shape is `(B, 20)`
- [ ] Plan and implement `train.py` — masked `BCEWithLogitsLoss` training loop, run on Colab L4
- [ ] Revisit distribution shift (training on full mixes vs. inferring on separated stems) before `evaluate.py`/`classifier.py` design