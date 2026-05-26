# Dev Journal — Day 6
**Date:** May 25, 2026
**Project:** Decomposition

---

## What We've Done Today

- Refactored `validation.py` and `processing.py` to use `pathlib.Path` instead of strings for consistency across the pipeline
- Wrote `separation.py` — the second stage of the pipeline
- Wrote full unit test suite for `separation.py`, all passing

---

## What We Built

**`separation.py` — 3 functions:**
- `separate(file_path, output_dir, model)` — public API, orchestrates the separation
- `_run_demucs(file_path, output_dir, model)` — private, builds and executes the Demucs CLI command via `subprocess.run`
- `_collect_stems(output_dir, track_name, model)` — private, constructs and returns the 4 stem paths as a dict

**Unit tests — `test_separation.py`:**
- `TestCollectStems` — pure path construction, no mocks needed
- `TestRunDemucs` — patches `subprocess.run`, tests success call, `CalledProcessError` → `DemucsFail`, `FileNotFoundError` → `DemucsNotFound`
- `TestSeparate` — patches `_run_demucs` and `_collect_stems` directly, asserts correct args and return value

---

## Key Concepts Learned

**No `--wav` flag in Demucs** — WAV is the default output format. Passing `--flac` or `--mp3` switches formats, but without either flag output is already WAV. The command only needs `-n`, `-o`, and the input path.

**`pathlib.Path.stem`** — returns the filename without the final extension. `Path("my_track.wav").stem` → `"my_track"`. Edge case: `"my_track.preview.wav"` → `"my_track.preview"` — only the last extension is stripped. Used to derive the track name for constructing Demucs output paths.

**`subprocess.CalledProcessError`** — requires `returncode` (int) and `cmd` (list) as positional args. To simulate in tests: `mock_run.side_effect = CalledProcessError(1, cmd)`.

**Private vs public functions** — `_run_demucs` and `_collect_stems` are prefixed with `_` because they're internal implementation details. Only `separate` is the public API. This means internals can change without breaking anything downstream.

**`@patch` decorator order** — decorators apply bottom-up, so parameters arrive in reverse order of the decorators. The bottom `@patch` maps to the first parameter.

**Convert at the boundary** — path type conversion (string → `Path`) should happen once at the entry point (`main.py`), not scattered throughout the pipeline. Every internal function speaks `Path` natively.

---

## Struggles

Knowing how to actually start writing a function from scratch is still hard — what the code should look like, what the first line even is. Getting better at breaking it down piece by piece (start with the docs) but this is an area to keep working on.

## Claude
I had claude summarize today and have it format the md... he's just a lot better at writing these than me

---

## Action Items for Next Session

- [ ] Smoke test `separation.py` on a real file from `data/test_audio/` — verify 4 stems appear in `data/separated/htdemucs/<track_name>/`
- [ ] Begin feature extraction stage — mel spectrogram, BPM, key using Librosa

---

## Remaining Roadmap

| Day | Stage |
|-----|-------|
| 7 | Feature extraction — mel spectrogram, BPM, key |
| 8 | CNN architecture research |
| 9 | OpenMIC data loader + weak label handling |
| 10 | CNN implementation + training loop |
| 11 | Training run + evaluation |
| 12 | JSON output + full pipeline integration in `main.py` |
| 13 | Buffer — bugs, cleanup, README, MVP done |