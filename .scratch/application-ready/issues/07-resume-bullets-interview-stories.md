Type: grilling
Status: open
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
