# Dev Journal — Day 14
**Date:** July 15, 2026
**Project:** Decomposition

---

## What We've Done Today

- Built the Colab training notebook from scratch, skeleton-first, same process as every other file
- Updated `config.py` to be environment-aware (Colab vs. local) via a `/content` existence check
- Updated `train.py` to import `CHECKPOINT_PATH` from `config.py` instead of duplicating it, and added GPU support (`.to(device)` for model and batch tensors)
- Kicked off the actual training run on Colab (L4 GPU)
- Finally resolved distribution shift — flagged and deferred since Day 9, revisited Day 11 and Day 12, closed out properly today

---

## Colab Notebook — Final Structure (5 cells)

1. **Clone repo** — clones from GitHub if not already present, `git pull` if it is; adds repo to `sys.path`
2. **Install dependencies** — `librosa`, `soundfile` only; Colab ships CUDA-enabled PyTorch already, no need to reinstall
3. **Mount Drive** — `drive.mount(...)` first, then `import src.config as config` (the *first* `src` import this session), asserts on `config.OPENMIC_DIR`, the npz file, and `config.TRAIN_PARTITION` existing; creates `CACHE_DIR`/`CHECKPOINT_PATH.parent` if missing
4. **Precompute mel cache** — `precompute_mel.main()`, prints cached `.npy` count as a sanity check before committing to training
5. **Train** — `train_mod.main()`, prints device, per-epoch loss, and final checkpoint confirmation

**Key ordering rule (the actual trap this design avoids):** Python caches imports in `sys.modules` by module name. The first time anything does `import src.config` (directly or transitively through `precompute_mel.py` or `train.py`), the `/content` check runs once and the resulting paths are locked in for the rest of the session — re-running the mount cell afterward does nothing to fix stale paths. Fix if this happens: restart the runtime. Cell 3 is deliberately the first cell to import `src.config`, and it does so only after `drive.mount(...)` has already run in the same cell.

**Why `config.py` handles Colab-vs-local instead of hardcoding paths in the notebook:** one source of truth. If `OPENMIC_DIR`/`CACHE_DIR`/`CHECKPOINT_PATH` were rebuilt by hand in both `config.py` and the notebook, changing the Drive folder layout later would mean updating two places instead of one. The notebook now only ever reads these values from `config`, never reconstructs them.

**GPU handling in `train.py`:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
...
spec, labels, label_mask = [t.to(device) for t in batch]
```
Device-agnostic — same file runs correctly on CPU (local) or GPU (Colab) with no manual toggling. Without this, PyTorch would default to CPU even with a GPU sitting right there unused, since nothing moves tensors onto it automatically.

**Deliberately deferred:** mid-run checkpointing (saving every N epochs, not just at the end). Current `train.py` only saves in `main()` after the full training loop returns — a Colab disconnect mid-run loses all progress back to epoch 0. Accepted as a conscious risk for this first run given time constraints; revisit only if a disconnect actually happens. Documented here specifically so it doesn't get silently forgotten.

---

## Distribution Shift — Finally Resolved

This has been flagged and deferred across Day 9, Day 11, Day 12, and the Day 13 handoff without ever being fully worked through. Closed out properly today via guided reasoning rather than just re-stating the bullet point again.

**The actual shift, stated precisely:**
- **Training distribution:** OpenMIC clips are full song mixes — vocals, drums, bass, and everything else, all playing together, exactly as released.
- **Inference distribution:** the classifier is fed the Demucs "other" stem — audio with vocals, drums, and bass **actively removed** by source separation, plus possible separation artifacts. The model has never seen this kind of stripped-down input during training.

**Key clarification from this session:** a standard train/test split (OpenMIC train partition vs. test partition) is *not* distribution shift — both come from the same underlying distribution, which is exactly what a train/test split is supposed to assume. Distribution shift is specifically about training data vs. real deployment-time data differing in some structural way. Important to keep these two concepts separate — conflating them would not hold up under a follow-up interview question.

**Why 4 of the 20 OpenMIC classes are "dead weight" at real inference (`bass`, `cymbals`, `drums`, `voice` — flagged originally on Day 9):**
- Demucs has already answered "is there voice / drums / bass present?" by construction — that's what separating into dedicated stems means. If a stem exists and isn't near-silent, the answer is yes.
- Asking the classifier to also predict those same 4 classes from the "other" stem is redundant at best — the "other" stem structurally cannot contain confident positives for them, since Demucs already extracted that content into other files.

**What this means in practice, split into two separate contexts:**
| Context | Include the 4 "dead" classes? |
|---|---|
| `evaluate.py` against OpenMIC's test set | **Yes** — measures the classifier's raw performance on the distribution it was trained on; all 20 classes are real, valid labels there |
| Real pipeline / `classifier.py` on Demucs "other" stems | **No** — these will almost never have true positives at inference; including them in a per-class report muddies the table and rewards "always predict absent" |

**Resolution for the MVP:** keep the 20-logit model head as-is (that's what OpenMIC trained, no need to retrain a 16-class version for v1). At the product/reporting layer, only surface the ~16 non-Demucs-covered classes from the CNN's output. Vocals/drums/bass presence gets reported separately, based on Demucs's own stem output (e.g. an energy/near-silence check on each stem) — not from the classifier at all.

**Flagged for later, not urgent:** the "is this stem silent" check for vocals/drums/bass presence needs its own logic when `classifier.py` gets built — separate entirely from the CNN, likely some kind of energy threshold on the stem's waveform.

---

## Key Concepts Learned

**Distribution shift vs. train/test split** — a train/test split assumes both sets come from the same distribution; that's the normal, expected condition for any supervised model. Distribution shift is specifically train-time data vs. real-world deployment-time data differing in a structural way that matters for the task.

**`sys.modules` import caching, applied to environment-detection config** — a module's top-level code runs exactly once per session, at first import, regardless of which file does the importing. This means environment-detection logic (like a `/content` check) must run *before* anything else imports that module, and re-running an earlier "setup" cell later does not undo an already-cached import. The fix if this state is reached is restarting the runtime, not re-running cells.

**Why GPU tensors don't move themselves** — `.to(device)` has to be called explicitly on both the model and every batch of data; PyTorch does not automatically use an available GPU just because one exists on the machine. Forgetting this means training silently runs on CPU even on a GPU-equipped Colab instance, with no error, just much slower training.

---

## Struggles

- Initially conflated a normal train/test split with distribution shift — needed to be walked back to what "distribution" actually refers to before the real issue (training-mix vs. inference-stem content) became clear
- First attempt at reasoning through the "dead weight" classes veered toward re-detecting voice/drums/bass presence *within* those dedicated stems, before landing on the cleaner insight that Demucs already answers that question by construction — the classifier doesn't need to re-solve an already-solved problem

---

## Action Items for Next Session

- [ ] Check on Colab training run — confirm loss trended downward across all 25 epochs, confirm checkpoint saved successfully to Drive
- [ ] Scope `evaluate.py` skeleton — file's one-sentence job, function signatures, per-class P/R/F1 design, now informed by the distribution-shift resolution (all 20 classes reported here, per the table above)
- [ ] When designing `classifier.py` later: build the "is this stem silent" check for vocals/drums/bass, and wire the ~16-class-only reporting logic for the "other" stem's CNN output

---

## Remaining Roadmap

| Stage | Status |
|---|---|
| `config.py`, preprocessing, separation, feature extraction, `dataset.py`, `model.py`, `train.py` | ✅ Done |
| Colab training run | ← **in progress** |
| `evaluate.py` | Next |
| `classifier.py` | Not started |
| `main.py` (pipeline wiring) | Not started |
| Full-stack web app | Post-MVP |
| Gemini benchmark comparison | Post-MVP, stretch goal / portfolio talking point |
