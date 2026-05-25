# Dev Journal - Day 5
**Date** May 24, 2026
**Project:** Decomposition

---

## What we've done today
- Finished and confirmed all preprocessing tests pass! All current logic is covered and green.
    - **Preprocessing tests:**
        - `TestNormalizeSampleRate` — confirms the audio gets resampled to 44.1 kHz when needed, and skips resampling if it’s already 44.1 kHz.
        - `TestNormalizeChannels` — verifies that 4-channel audio is downmixed correctly to stereo, and that stereo (2-channel) audio remains unchanged.
        - `TestProcessAudio` — checks the overall pipeline: both resampling and channel normalization, plus ensuring files are written to the correct `data/processed` path with correct output sample rate and shape.
    - **Validation tests:**
        - Covered all validation rules for `validAudio`, including extension checks (accepts `.wav`/`.WAV`, rejects `.mp3` etc.), corrupted file detection (using librosa mock failures), and precise duration bounds (raises on <5s or >600s, passes valid range).
        - Confirmed exceptions are handled as expected: `IncorrectExtension`, `CorruptedFile`, and `InvalidLength`.
        - Used unittest patching and MagicMock to isolate logic and simulate file scenarios — no real audio files needed for tests.
- All tests are passing as of today. Confidence in preprocessing is now high and bugs are much less likely to sneak through.

- Researched the 2 ways to call Demucs (CLI vs API) and chose CLI since we're doing one song at a time, already saving to disk

- Designed separation.py — 3 functions: run Demucs, collect stem paths, main separate function that ties them together
- Updated folder structure — added data/separated/ for stem outputs

data/
├── openmic/
├── test_audio/
├── processed/
└── separated/


## Key Concepts Learned

#### CLI

**Pros:**
- Demucs runs in its own process, so it cleans up memory itself.
- Manages PyTorch audio tensors and custom data shapes internally.
- Less work for us.
- Automatic file handling: decodes, chunks, splits, and writes output files.

**Cons:**
- Heavy overhead — each call reloads the large model weights.
- Must save audio to disk (which we already do, so this is acceptable).

#### API

**Pros:**
- Model weights are loaded once into memory, so you can process many audio snippets with no repeated startup lag.
- Would be useful if we want to train our classifier on the "other" stem.
- Can work with real-time audio input (not needed now, but could be a cool implementation for the future).
- Lets you change hyperparameters, e.g. to avoid CUDA out-of-memory errors.

**Cons:**
- Requires us to manage PyTorch, torchaudio, and other dependencies.
- If not handled carefully, can exhaust system RAM or VRAM.

- structure for file
    - Take the processed WAV path as input
    - Call Demucs on it
    - Capture the 4 output stem files (vocals, drums, bass, other)
    - Return their paths for the next pipeline stage

- To Capture Output (Hide from Terminal):
    - Set capture_output=True. This silences the terminal and saves the data into result.stdout and result.stderr.

- for catching exceptions 
    - subprocess.CalledProcessError
    - FileNotFoundError checks if demucs command was not found, but the calledprocesserror just works if demucs started but failed with exit code




## Action Items for Next Session
- write and test the separation file and maybe research for the next stage in pipeline