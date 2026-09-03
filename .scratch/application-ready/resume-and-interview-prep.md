# Resume bullets + interview stories

Closes ticket 07. Every number here is measured and traceable to the README
or `docs/research/`. Nothing is rounded up, and nothing is claimed that a
follow-up question would collapse.

---

## Where I disagree with the ticket

The ticket's instinct was to build all three stories out of the three
measured failures: the stem-recalibration non-replication, the aggregation
claim walked back, and the grounding hole the eval harness found.

Failures-that-got-measured are the right *material* — that instinct is
correct and it's what separates this from a tutorial project. But all three
of those are the **same story archetype**: "I believed X, I measured X, X was
false, I said so." An interviewer hears one signal three times and learns
nothing new on the second and third telling.

Two more specific problems:

- **The aggregation walk-back is weak as a standalone story.** What went
  wrong there was a *claim in conversation*, not an engineering decision.
  Stripped to its facts it's "I ran an ablation and it came out neutral"
  (BEATs 0.555 → 0.562, CNN 0.513 → 0.509). That's a good supporting detail,
  not a five-minute answer. It lives inside Story 3 below.
- **Nothing in that set shows you building something that worked.** Three
  failure stories in a row reads as a candidate whose project didn't land.
  It did land — there's a live URL.

So: keep the failures, but distribute them. Each story below is built around
a decision that produced a result, and each one has a measured failure inside
it. Story 2 in particular is the highest-signal item in the whole project for
AI-engineering roles, because most student "AI projects" are a prompt and a
fetch call, and this one is a system that declines to answer.

Story 4 is a reserve for SWE/full-stack-leaning interviews.

---

## Resume bullets

**Decomposition** — audio stem separation + instrument recognition + grounded
chat agent · Python, PyTorch, Next.js/TypeScript · [live](https://decomposition-three.vercel.app) · [github](https://github.com/alexnguyen25/Decomposition)

- Improved multi-label instrument-recognition macro-F1 from **0.650 to 0.805**
  on OpenMIC-2018 (20 classes, 5,085-clip test set) by benchmarking 8
  approaches from a scratch CNN through VGGish and PANNs to BEATs; shipped a
  **1.6 MB** MLP head on frozen BEATs embeddings after end-to-end fine-tuning
  of the backbone (62 min) failed to beat a head trained in **12 seconds**.
- Quantified the benchmark-to-deployment gap instead of hiding it: the same
  model scores **0.805 on 10-second benchmark clips but 0.562 on 14 real
  full-length songs** run through Demucs separation, and both numbers are
  published in the README with the methodology.
- Built a **grounded chat agent** (Next.js route handler, OpenAI-compatible
  tool calling) that can only answer from classifier output, via a
  tool-call → post-hoc validation → retry → refuse contract; a 30-question
  eval harness measures the hallucination rate and caught a real hole where
  generic stem words were allow-listed regardless of whether the stem existed.
- Shipped the full stack: Demucs + BEATs Python pipeline, FastAPI service,
  Next.js frontend with four-stem synchronised playback, deployed free on
  Vercel with precomputed analyses served as static assets; CI runs pytest,
  ruff and `next build` on every push.

### Notes on the bullets

- The ticket asked whether the chat agent deserves its own line. It does.
  "Trained a classifier" and "built an agent that provably doesn't make things
  up" are different skills, and a lot of teams are hiring specifically for the
  second one right now.
- Bullet 2 is the one most likely to get you asked a real question. Most
  candidates only put the flattering number on the page. Leading with the gap
  is a deliberate filter for teams that care about rigour — and if a team
  reads it as weakness, that's information about the team.
- **0.805, not 0.8045.** Three significant figures on a resume; the exact
  figure is in the README if anyone checks.

---

## Story 1 — "Everything I trained from scratch lost to a frozen encoder"

**The question it answers:** _Tell me about a technical decision you made with
data._

The baseline was a 2-conv CNN trained from scratch on OpenMIC-2018, scoring
0.650 masked macro-F1. The obvious next move is a bigger CNN, and I did that
first — 4 conv blocks got to 0.699. But OpenMIC is only ~15k weakly labelled
clips, which is not enough to learn general audio representations from
nothing, so I switched from training features to reusing them: VGGish (0.746),
then PANNs CNN14 (0.792), then BEATs (0.805 with tuned thresholds).

The decision I actually care about is what I did next. The natural instinct
was to fine-tune the pretrained backbone end-to-end. I ran it: 62 minutes of
training on CNN14 produced 0.7905, against 0.7915 for the *same frozen
embeddings* with a head that trained in 12 seconds. Statistically identical,
300× the compute.

So the shipped model has a frozen backbone and a 1.6 MB head. That's not a
compromise, it's the measured result — and it's also why the whole thing can
be a 1.6 MB file in git with the 345 MB backbone pulled from the HF Hub.

**Failures inside this story:** SpecAugment hurt (0.699 → 0.684 on the 4-conv
CNN). Ensembling the two best models scored 0.7996, *below* BEATs alone at
0.8045 — the weaker model dilutes the stronger.

**What I'd do differently:** I tuned decision thresholds per class, which is
worth ~1 point on BEATs (0.7921 → 0.8045) but was worth 10 points on the
baseline CNN re-run (0.5638 → 0.6701). I should have caught earlier that a
big threshold-tuning gain is a symptom of a poorly calibrated model, not a
free win.

**Likely follow-ups:**
- _Why masked macro-F1?_ OpenMIC labels only a handful of the 20 classes per
  clip; the rest are genuinely unknown, not negative. Scoring unknowns as
  negatives would reward a model for staying quiet. The mask means a class
  only contributes to the metric on clips where a human confirmed it either
  way.
- _Why macro not micro?_ The classes are very imbalanced. Micro-F1 would let
  the model win by getting guitar and drums right and ignoring accordion.
- _Isn't 0.650 → 0.805 partly just a better baseline comparison?_ Yes, and
  that's in the README. The committed 0.650 checkpoint was trained without a
  validation split. The same recipe re-run properly scores 0.5638 @0.5 and
  0.6701 tuned. I report 0.650 as the honest starting point because it's the
  number that was actually committed, and I show both rows.

---

## Story 2 — "I built an agent whose main feature is refusing to answer"

**The question it answers:** _Tell me about something you built that involved
an LLM._ / _How do you handle hallucination?_

The app lets you ask questions about a track. The failure mode that would
destroy the whole point is the model confidently describing a saxophone solo
that the classifier never detected — because then the chat is just a language
model riffing on a filename, and the classifier I spent weeks on is
decoration.

So the agent never sees the audio and has no free-form knowledge of the track.
It gets three tools — `get_instruments`, `get_stem_activity`, `get_bpm_key` —
that read the structured analysis, and the contract is:

1. the model calls tools;
2. the draft answer is validated **after generation** against the tool
   results, on word boundaries, with presence flags consulted;
3. if it asserts something not in the tool output, it gets one retry with the
   violation quoted back to it;
4. if it fails again, the app returns a refusal instead of the answer.

Post-hoc validation is the part I'd defend hardest. Prompting a model to
"only use the tools" is a request; validating the output is a guarantee. The
system prompt can be ignored, the validator cannot.

**The failure inside this story, and it's the good part:** I wrote a
30-question eval harness — factual, temporal, out-of-scope, and "trap"
questions that ask leading questions about instruments that aren't there
("how prominent is the saxophone?"). The harness caught that generic stem
words — `bass`, `vocals`, `drums` — were on an unconditional allow-list, so
the agent could assert "there's a bass line under the chorus" about a track
whose bass stem is *silent*. It was allow-listed because those words are
usually fine. The fix was to gate them on the actual presence flags. I would
not have found that by reading the code; I found it because I wrote adversarial
questions and ran them.

**Second failure worth telling:** an early harness run reported a perfect
score when in fact all 30 requests had been rate-limited and failed — it was
scoring error strings as non-hallucinations. A harness that reports success
when nothing ran is worse than no harness, because you believe it. It now
refuses to summarise any run containing failures.

**Likely follow-ups:**
- _Why not just use a bigger model?_ It doesn't address the failure mode. A
  bigger model hallucinates less often, not never, and "less often" is not a
  property you can put in a README. The validator's guarantee is independent
  of the model, which is also why swapping providers is two env vars.
- _What does validation cost?_ One extra generation on the retry path only.
  The validator itself is string matching over a small set of names.
- _False refusals?_ That's the real tradeoff, and it's the thing to watch. The
  presence-gating fix made the validator stricter, which by construction can
  only increase refusals. The out-of-scope and factual buckets in the eval are
  what keep that honest.

---

## Story 3 — "The benchmark number was not the real number"

**The question it answers:** _Tell me about a time you were wrong._ / _How do
you know your model works?_

BEATs scores 0.805 on the OpenMIC test set. The product doesn't see OpenMIC.
It sees the "other" stem that Demucs produced from a whole song — different
length, different spectral content, artefacts from the separation. So I built
a second evaluation on 14 CC-licensed MTG-Jamendo tracks and measured there
too: **0.562**. That's the number that describes the product.

Then I tried to close the gap. On PANNs, fine-tuning the head on
Demucs-processed clips had lifted real-song recall from 0.585 to 0.659 — a
real, useful effect. So I applied the same recipe to BEATs, and expected the
same lift.

It did not replicate. Measured over 19 real tracks, the recalibrated head and
the standard head scored a mean F1 of **0.524 and 0.524** — indistinguishable.

The tempting move was to quietly ship the recalibrated head anyway, since it
was no worse and the story was nicer. Instead the recalibrated head ships
behind a `CLASSIFIER_HEAD=stem_recalib` switch and is written up in the README
as a negative result. A technique that works on one backbone is evidence about
that backbone, not a law.

**Supporting detail — the smaller thing I also got wrong:** I expected
switching chunk aggregation from max-pool to mean-of-top-3 to be a
significant fix for over-prediction. Measured, it's roughly F1-neutral: BEATs
0.555 → 0.562, and the CNN actually gets *worse*, 0.513 → 0.509. It ships
anyway, because it trades recall for precision and a wrong instrument is
visible on screen while a missing one isn't — but it ships as a product
judgement, not as a performance fix, and the README says so.

**Likely follow-ups:**
- _Is 14 tracks enough to conclude anything?_ No, and that's stated. It's
  enough to establish that the gap is large and real; it is not enough to rank
  two heads that differ by 0.000. The honest conclusion from n=19 is "no
  detectable difference", which is exactly what I wrote.
- _Why is precision a lower bound?_ Jamendo tags are weak and incomplete. An
  instrument that's genuinely playing but untagged counts as a false positive.
- _So which number do you believe?_ 0.562, for this product. 0.805 is the
  number that lets me compare against other people's OpenMIC results.

---

## Story 4 — reserve, for SWE / full-stack interviews

**"Two bugs lived through a green test suite for two weeks."**

The preprocessing tests mocked `soundfile.write`. Everything passed. Two real
bugs sat underneath the mock the whole time — including one where the
subprocess call used a literal `"python3"`, which resolved to a Homebrew
interpreter that didn't have Demucs installed, and the `CalledProcessError`
handler discarded Demucs' stderr, so the failure surfaced as a bare exit code
with no message.

Mocking an I/O boundary tests the logic *around* the boundary and asserts
nothing about whether the plumbing works. There's now an unmocked round-trip
test that writes a real temporary file, the subprocess uses `sys.executable`
so it inherits the running interpreter, and the error path puts the last 800
characters of stderr into the raised exception.

The generalisable point: a test that mocks the thing most likely to break is
measuring your confidence, not your code.

---

## Rehearsal notes

- Lead with the number, then the decision, then the caveat. Not the reverse.
- When you say 0.805, say 0.562 in the same breath. If an interviewer has to
  drag the weaker number out of you, you've lost more than the number was
  worth.
- "I expected X, I measured X, X was false, here's what shipped" is the shape
  of all four. That shape is the point.
- Don't say "we" — this is a solo project and it's stronger as one.
