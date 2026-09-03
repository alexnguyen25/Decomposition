Type: grilling
Status: resolved (2026-09-03)

## Question

`improve-f1` and `main` currently point at the same commit; everything from
the last two research phases — `docs/research/`, `data/f1_research/`
(checkpoints, caches, the HF Space folder), and all of `reference-app/` — is
untracked. What's the actual commit/gitignore/LFS strategy before any of
this gets pushed?

Needs a real decision on:
- Which large binaries (the 352 MB BEATs checkpoint, other `.pt` files,
  cached embeddings/probs) go in Git LFS vs. get gitignored entirely (e.g.
  regenerable caches, `.npz`/`.npy` intermediates) vs. live outside the repo
  with a documented fetch step.
- Whether `docs/research/` and `data/f1_research/`'s scripts/results (not
  the checkpoints) get committed as-is, given they're part of the resume
  story ("show your work" per the §12 checklist).
- Whether `reference-app/` gets committed now (planning-only per this map,
  but the commit strategy still needs deciding for when Alex is ready) or
  waits until it's closer to deployable.
- Whether to clean up commit history on `improve-f1` before merging to
  `main`, or just commit forward from here.

## Resolution (2026-09-03)

Heads (1.6 MB each) commit to git; the 345 MB BEATs backbone lives at
`alexnguyen25/decomposition-models` on the HF Hub and is fetched by
`scripts/fetch_models.py`; regenerable research blobs (.npz/.npy/jamendo/
pylibs) gitignored, scripts and JSON results committed. No Git LFS.
Result: 180 files / 28.5 MB would commit.
