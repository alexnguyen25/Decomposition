Type: grilling
Status: resolved (2026-09-03)
Blocked by: 02

## Question

The MVP research doc (§12) already gives an ordered, research-backed
checklist for the README (live demo URL, GIF, metrics table, research-log
section including failures, one-command quickstart, limitations section).
What's undecided is the actual content once real numbers exist:

- Which metrics headline the table — does it use the src/-integrated
  BEATs result (pending ticket 02) or the data/f1_research number, and does
  the chat agent's measured hallucination rate (0% on the 30-question eval
  harness, llama3.2:3b) get a place in the headline or stays in a deeper
  section?
- Exact framing of the classifier-vs-Gemini comparison (masked macro-F1,
  per-instrument phrasing per the existing "claim only measured facts"
  guidance) — does it get rerun with whatever's actually shipped in
  production (ticket 03's provider, ticket 02's integrated head) or does it
  stay a research-artifact number with a caveat?
- What goes in "failures/research log" — SpecAugment hurting, max-pooling
  false-positive explosion, and now presumably the chat agent's
  presence-gate fix (the eval harness catching the "faint bass activity"
  false-positive from peak-normalized envelopes) as a fresh example.

Blocked on ticket 02 because the headline metric needs to be the real,
integrated `src/` number, not the research-scratchpad one.

## Resolution (2026-09-03)

README written with the shipped numbers: 0.6498 -> 0.8045 OpenMIC,
0.562 real-song F1, 0.8517 vs Gemini 0.5981, 0/30 hallucinations. Research
log includes the failures. Only the live URL is still a placeholder.
