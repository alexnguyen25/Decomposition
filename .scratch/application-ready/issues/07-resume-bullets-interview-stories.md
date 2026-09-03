Type: grilling
Status: closed
Blocked by: 06

## Question

The MVP doc has a draft bullet ("Improved instrument-classification macro-F1
0.650→0.794 on OpenMIC-2018 by benchmarking 8 approaches...") written before
BEATs was integrated into `src/` and before the chat agent existed. Needs a
real pass once the README's final numbers are locked:

- Final bullet phrasing — does it now read 0.650→0.8045 (or whatever the
  `src/`-integrated number turns out to be), and does a second bullet cover
  the chat agent (grounded tool-use, measured hallucination rate) as its
  own line, given "AI engineering" is a distinct skill signal from "ML
  research"?
- Which 3 interview stories get rehearsed — the MVP doc suggested
  PANNs-vs-scratch-CNN trade-off, the SpecAugment/max-pooling failure, and
  the masked-macro-F1 measurement decision. Does the chat agent's grounding
  contract (validate → retry → refuse, never silently hallucinate) replace
  one of these, or add a 4th?

Blocked on the README ticket because bullets should quote numbers that are
already locked and public, not numbers that might still shift.

---

## Resolution (2026-09-03)

Answered in `.scratch/application-ready/resume-and-interview-prep.md`.

- Bullet reads **0.650 → 0.805** (3 s.f.; exact 0.8045 stays in the README).
- The chat agent gets its own bullet. "Trained a classifier" and "built an
  agent that provably doesn't fabricate" are different hiring signals.
- Added a bullet the ticket didn't anticipate: the benchmark-to-deployment
  gap (0.805 vs 0.562), stated up front rather than buried.
- **Stories: partly disagreed with the ticket's framing.** Three
  failure-stories share one archetype and an interviewer learns nothing on
  the second telling; the aggregation walk-back is too thin to carry five
  minutes on its own. Final three are frozen-embeddings, the grounding
  contract, and the generalisation gap — each built around a decision that
  produced a result, each containing a measured failure. The aggregation
  result is folded into story 3; the mocked-I/O-boundary bug is story 4, held
  in reserve for SWE-leaning interviews. Reasoning is written out in the
  deliverable.
