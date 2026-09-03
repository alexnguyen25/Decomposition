Type: grilling
Status: resolved (2026-09-03)

## Question

The BEATs-embeddings + MLP head classifier (frozen BEATs, 0.8045 tuned
macro-F1, proven in `data/f1_research/`) needs to move into `src/` as Alex's
own tutored implementation, replacing or supplementing the current tiny
2-conv CNN in `src/classification/model.py` (trained on OpenMIC mel
spectrograms, baseline macro-F1 0.650).

Needs a real decision on:
- Replace the CNN outright, or keep both with a config switch (and if kept,
  why — is the CNN's 0.650 baseline itself part of the resume story, i.e.
  "here's what I tried first")?
- Where does BEATs (a 352 MB pretrained checkpoint) live relative to
  `src/` — vendored, downloaded on setup, or referenced from
  `data/f1_research/`?
- Does the stem-domain fine-tuned head (`ckpt_E10_stem_recalib.pt` —
  better real-song recall) get included from the start, or is that a
  second pass after the base head lands?
- What does Alex need to understand and reason through himself vs. what's
  pure plumbing (loading a pretrained checkpoint) — this shapes how the
  tutoring session should be paced.

This ticket is the tutoring session itself (grilling + domain-modeling,
Socratic, Alex reasons to the design) — not a spec Claude writes and hands
over.

## Resolution (2026-09-03)

BEATs vendored at `src/classification/beats/` (MIT, NOTICE.md included);
`embedder.py` + `head.py` added; `classifier.py` dispatches on `BACKEND`
so the 0.650 CNN stays as the documented baseline. Shipped head is `e10`,
not `stem_recalib` — they are statistically identical on real songs and
`e10` is the reproducible one. `scripts/eval_openmic.py` reproduces 0.8045.
