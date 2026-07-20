# Dev Journal — Day 17
**Date:** July 19, 2026
**Project:** Decomposition

---

## What We've Done Today

- Designed and wrote `main.py` — the top-level pipeline wiring, skeleton-first, same process as every other file
- Added `check_stem_presence` to `stem_presence.py` — the missing Path→waveform glue flagged (but not built) on Day 16
- Ran the full pipeline end-to-end for the first time — real audio in, real structured JSON out
- Found and fixed two real bugs in `processing.py`, both hidden by mocked tests until this run
- **MVP is done.** Preprocessing → separation → feature extraction → stem presence → classification → structured JSON, all wired, all running.

---

## What We Built

**`main.py` — three functions, matching the `train.py`/`evaluate.py` pattern:**
- `setup() -> tuple[Model, torch.device]` — picks device, loads the trained checkpoint once per process
- `analyze(file_path: Path, model: Model, device: torch.device) -> dict` — the actual per-song pipeline: validate → preprocess → separate → extract features → check stem presence → classify → assemble result dict
- `main(file_path: Path) -> dict` — thin orchestrator, calls `setup()` then `analyze()`; the `if __name__` block owns `sys.argv` parsing and `json.dumps` printing, kept separate so `main()` itself stays callable and testable

**`stem_presence.py` addition:**
- `check_stem_presence(stems: Mapping[str, Path]) -> dict[str, bool]` — loads each stem's waveform with `librosa.load(path, sr=None, mono=False)`, calls `is_stem_silent`, negates to get "present," builds the result dict. Lives next to `is_stem_silent` since it's the same audio I/O concern, not a `main.py` concern.

**Final JSON shape (locked, verified against a real run):**
```json
{
  "bpm": 132.51,
  "key": "C# minor",
  "stems": { "vocals": "...", "drums": "...", "bass": "...", "other": "..." },
  "presence": { "vocals": true, "drums": true, "bass": true },
  "instruments": [
    { "name": "guitar", "confidence": 0.734 },
    { "name": "saxophone", "confidence": 0.567 },
    { "name": "trumpet", "confidence": 0.563 }
  ]
}
```

---

## Key Design Decisions

**`setup()` / `analyze()` split, not one big function.** Motivation is different from `evaluate.py`'s split (which was about keeping `compute_metrics` pure and testable). Here it's about lifecycle scope: model weights are process-scoped (expensive, load once), analyzing a song is request-scoped (cheap, runs per file). Without the split, a future batch CLI or web handler would either reload weights per request or require tearing `main()` apart under pressure. `analyze()` is the reusable unit; `main()` is the single-song convenience entry point.

**Dict shape is stage-aligned, not flat and not per-stem-nested.** Each top-level key (`bpm`, `key`, `stems`, `presence`, `instruments`) maps ~1:1 onto one pipeline stage's return value. Rejected a fully flat shape (scatters related fields) and a per-stem-object shape (`stems.vocals: {path, present}`) because that forces asymmetric shapes across stems — `other` would need an `instruments` key where the rest have a `present` key. The chosen shape means assembling the dict in `analyze` is mostly "put each stage's return value under its key," not inventing a second schema.

**`check_stem_presence` takes exactly 3 stems in, 3 out — not all 4 with `other` skipped internally.** Considered accepting the full 4-stem dict and skipping `other` inside the function, rejected it: a 4-in signature would silently promise more than the body delivers, hiding a pipeline-level decision (why `other` is excluded) inside a helper that has no business making that call. The three-key contract makes scope legible from the signature alone, and forecloses someone later "completing" it to check all four stems, which would quietly reactivate the placeholder `"other": 0.01` entry in `STEM_FLOORS`.

**Error handling: `analyze` doesn't catch `validAudio`'s exceptions.** How a failure should surface (CLI: stderr + nonzero exit; future web layer: HTTP 400) is a caller decision, and differs per caller. Catching in `analyze` would force one presentation choice on every future caller of the pipeline.

**`separate()`'s output directory nesting stays fully internal to `separate()`.** Traced through `_run_demucs`/`_collect_stems` to confirm `main.py` never needs to know about the `htdemucs/<track_name>/` folder structure — `SEPARATED_DIR` passed in is just the base, and the returned dict already contains the correct deep paths. Verified this wasn't a silent-wrong-path risk before trusting it.

---

## Bugs Found During the First Real Run

Both bugs were sitting in `processing.py` since Day 3, both invisible to `test_preprocessing.py` because the tests mock `sf.write` directly — meaning the tests confirmed "we called `sf.write` with roughly the right shape of arguments," never that writing a file actually works.

1. `sf.write(..., y=y, ...)` — wrong keyword argument; should be `data=`.
2. `project_root` was resolved one `dirname` level too shallow, so output was written to a nonexistent `src/data/processed/` instead of the repo-root `data/processed/`. Fixed the path resolution and added a defensive `os.makedirs`.

**The real lesson here, worth keeping:** unit tests with a mocked I/O boundary prove the surrounding *logic* is correct, not that the *plumbing* underneath actually works. Only an unmocked, end-to-end run exercises the real call and can catch this class of bug. Same category of lesson as the eval-run stall (Day 15) and the over-prediction discovery (Day 16) — the pattern of "the real run surfaces something the design-on-paper couldn't" keeps recurring, and it's a reason to run real end-to-end passes early and often rather than trusting a fully-green mocked test suite as sufficient.

---

## Known Naming Artifact (flagged, not fixed)

`separate()`'s `track_name` derives from `processed_path.stem`, and `main.py` passes the *processed* file (not the original) into `separate()`. Since the processed filename has `_processed` appended, the Demucs output folder — and therefore every stem path in the final JSON — reads `<name>_processed/` instead of the original track name. Internally consistent (no path drift, nothing breaks), purely cosmetic. Left as a conscious, named decision rather than patched reflexively; revisit only if the original filename needs to surface in product-facing output later (e.g. the web app).

---

## Struggles

- None significant this session — the skeleton had already been fully reasoned through function-by-function in the prior session (signatures, dict shape, three-in/three-out contract, error propagation all decided before any code was written), so writing the actual implementation was mostly translation rather than discovery.
- The two `processing.py` bugs weren't really "struggles" so much as confirmation that a real run finds things a paper design can't.

---

## Current Project State

**Built, verified, and reviewed:**
- `src/config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `classifier.py`, `stem_presence.py` — all prior sessions
- **`src/separation/stem_presence.py` — `check_stem_presence` added this session**
- **`main.py` ✅ NEW — fully written, reviewed, and run end-to-end this session**

**MVP status: complete.** One local audio file path in → structured JSON out (BPM, key, stem paths, vocals/drums/bass presence, "other"-stem instrument list) → stems also written to disk.

**Still open, deliberately deferred:**
- Day 16's over-prediction problem (false positives clustering in the wind/brass/string family) — not addressed this session, next natural focus now that real end-to-end JSON output exists across potentially multiple songs
- `stem_presence.py`'s floor/fraction constants — still placeholder, calibrated against only one track
- Naming artifact above (`_processed` leaking into stem folder names)
- Whether to add a real (non-mocked) regression test for the two `processing.py` bugs just fixed, or rely on `main.py`'s end-to-end run as ongoing coverage — undecided, flagged explicitly so it isn't dropped by accident

**Not yet built (post-MVP):**
- Full-stack web app
- Gemini benchmark comparison

---

## Action Items for Next Session

- [ ] Return to the Day 16 over-prediction problem — now with a real end-to-end JSON pipeline, run `analyze()` against 2–3 more real songs with known instrumentation and look at raw pre-threshold probabilities across all 20 classes, not just what clears 0.5
- [ ] Decide on the mocked-vs-real test coverage question for `processing.py` flagged above
- [ ] Revisit the `_processed` naming artifact if/when it matters for a real user-facing surface

---

## Remaining Roadmap

| Stage | Status |
|---|---|
| `config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py`, `train.py`, `evaluate.py`, `classifier.py`, `stem_presence.py` | ✅ Done |
| `main.py` (pipeline wiring) | ✅ Done — this session |
| **MVP** | ✅ **Complete** |
| Diagnose over-prediction issue | Next |
| Full-stack web app | Post-MVP |
| Gemini benchmark comparison | Post-MVP, stretch goal / portfolio talking point |
