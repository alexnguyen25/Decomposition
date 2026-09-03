# Decomposition

**Pull a song apart and ask it questions.** Upload audio, get four isolated
stems, the instruments playing inside them, tempo and key — then chat with an
agent that can only answer from what the model actually measured.

> **Live demo:** _(add your Vercel URL here after deploying — see [DEPLOY.md](DEPLOY.md))_

Separation is [Demucs](https://github.com/facebookresearch/demucs). Instrument
recognition is a classifier I trained on [OpenMIC-2018](https://zenodo.org/record/1432913):
frozen [BEATs](https://arxiv.org/abs/2212.09058) embeddings plus a 20-way MLP head,
**macro-F1 0.8045**, up from **0.650** for the scratch CNN I started with.

---

## What it does

| | |
|---|---|
| **Stem separation** | vocals / drums / bass / other, playable and downloadable |
| **Instrument recognition** | 20 classes, per-song and per-10-second |
| **Tempo & key** | librosa beat tracking, Essentia key detection |
| **Grounded chat** | ask about the track; every fact comes from a tool reading the analysis |

## Results

### Instrument classification (OpenMIC-2018 test split, masked macro-F1)

| Model | @0.5 | tuned thresholds |
|---|---|---|
| Scratch 2-conv CNN — the committed baseline, no val split | **0.6498** | — |
| Same recipe re-run with a validation split | 0.5638 | 0.6701 |
| 4-conv CNN | 0.6989 | 0.7135 |
| VGGish + head | 0.7462 | 0.7518 |
| PANNs CNN14 + head | 0.7891 | 0.7915 |
| CNN14 fine-tuned end-to-end | 0.7874 | 0.7905 |
| **Frozen BEATs + MLP head — shipped** | **0.7921** | **0.8045** |

The two baseline rows are the same architecture measured two ways: the
original checkpoint trained without a validation split (0.6498, the number the
0.650 → 0.8045 headline starts from), and the same recipe re-run inside the
experiment harness with a held-out split for honest threshold tuning.

Reproduce: `python scripts/eval_openmic.py`. It re-derives the number from the
committed head and checks that this repo's embedder still produces the exact
embeddings the head was trained against.

### Real songs (14 CC-licensed MTG-Jamendo tracks, Demucs "other" stems)

OpenMIC clips are 10 seconds of full mix; the product sees a separated stem
from a whole song. That gap is real, so it gets measured separately:

| Backend | precision | recall | F1 | predictions/track |
|---|---|---|---|---|
| Scratch CNN | 0.427 | 0.707 | 0.509 | 9.9 |
| **BEATs + head** | **0.595** | 0.627 | **0.562** | **5.7** |

Reproduce: `python scripts/eval_real_songs.py`. Jamendo tags are weak and
incomplete — an instrument that is genuinely present but untagged counts here
as a false positive, so precision is a lower bound.

### Versus a frontier multimodal model (100 OpenMIC test clips)

Gemini gets the raw audio and is asked to name the instruments; the classifier
gets the same clips. Both are scored as a set of names per clip, masked
macro-F1:

| | macro-F1 |
|---|---|
| **BEATs + head (shipped), tuned thresholds** | **0.8517** |
| BEATs + head, @0.5 | 0.8515 |
| `gemini-flash-latest` | 0.5981 |

Reproduce: `python scripts/eval_vs_gemini.py` — **no API calls**, it re-scores
Gemini's cached per-clip predictions from the July benchmark against a
classifier arm computed fresh from `src/`.

Read honestly, this is not "the model is better than Gemini". It is the
narrower and more useful claim that a **1.6 MB task-specific head on frozen
embeddings beats a general frontier model at the one thing it was trained for**,
at a fraction of the cost and latency. Two caveats worth stating: 100 clips is a
small sample — PANNs scored 0.8751 on this same subset despite losing to BEATs
on the full 5,085-clip test set (0.7915 vs 0.8045), which is sample noise, not a
real ordering. And `gemini-pro-latest` only ever ran on 50 of the 100 clips
(0.569 on its own subset), so it is left out rather than compared unevenly.

### Chat agent grounding (30 scripted questions)

| | |
|---|---|
| Hallucination rate | **0/30** |
| Factual · temporal · trap · out-of-scope | 12/12 · 6/6 · 6/6 · 6/6 |
| Mean latency | 2.7 s (local `llama3.2:3b`) |

Reproduce: `python evals/run_evals_http.py --base-url http://localhost:3002`.
"Trap" questions ask leading questions about instruments that aren't there
("how prominent is the saxophone?"); passing means the agent declines to play
along.

---

## Architecture

```
                 ┌──────────────────────── Vercel ────────────────────────┐
  browser ──────▶│  Next.js                                               │
                 │    /                     landing + example shelf       │
                 │    /track/[id]           waveforms, stems, instruments │
                 │    /api/chat/[id]        grounded agent (TypeScript)   │
                 │    /examples/*           precomputed JSON + stem mp3s  │
                 └────────────────────────────┬───────────────────────────┘
                                              │ OpenAI-compatible
                                              ▼
                                        Groq  llama-3.1-8b-instant

  ── local only, too heavy for free serverless ──────────────────────────
  audio ─▶ validate ─▶ resample/downmix ─▶ Demucs ─▶ ┬─ vocals ┐
                                                     ├─ drums  ├─ silence check
                                                     ├─ bass   ┘
                                                     └─ other ──▶ BEATs ──▶ MLP head
                                                                            │
                                          librosa BPM + Essentia key ───────┤
                                                                            ▼
                                                                    structured JSON
```

The deployed site serves **precomputed** analyses as static files, so it needs
no server to show a result and never cold-starts. Analysing your own audio runs
Demucs and a 345 MB model — minutes of CPU and several GB of RAM — which no free
serverless tier provides, so that part runs locally.

## Quickstart

```bash
git clone https://github.com/alexnguyen25/Decomposition
cd Decomposition
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_models.py          # 345 MB BEATs backbone from the HF Hub
python -m src.main path/to/song.mp3
```

The trained heads are committed (they are small, and they are the part that was
actually trained here). Only the third-party BEATs backbone is downloaded, from
[alexnguyen25/decomposition-models](https://huggingface.co/alexnguyen25/decomposition-models).

Run the web app locally, including uploads:

```bash
cd reference-app/backend && ../../.venv/bin/uvicorn app:app --port 8000
```

```bash
cd reference-app/frontend && BACKEND_ORIGIN=http://localhost:8000 NEXT_PUBLIC_UPLOAD_ENABLED=1 npm run dev
```

## Research log

Eight experiments, written up in [`docs/research/`](docs/research/), including
the ones that didn't work — those were the informative ones.

- **Frozen pretrained embeddings beat everything trainable from 15k weakly
  labelled clips.** BEATs 0.8045 and PANNs 0.7915 against 0.650 for a scratch CNN.
- **Fine-tuning the backbone end-to-end was a waste.** CNN14 fine-tuned for 62
  minutes (0.7905) *tied* the same embeddings with a head trained in 12 seconds
  (0.7915). Negative result, and the reason the shipped backbone is frozen.
- **SpecAugment hurt.** 0.6989 → 0.6844 on the 4-conv CNN.
- **Ensembling the two best models underperformed the better one alone**
  (0.7996 vs 0.8045) — the weaker model dilutes the stronger.
- **Stem-domain recalibration did not replicate.** Fine-tuning the head on
  Demucs-processed clips lifted real-song recall on PANNs (0.585 → 0.659), so
  the same recipe was applied to BEATs — but measured on 19 real tracks the two
  heads are indistinguishable (mean F1 0.524 vs 0.524). The recalibrated head
  ships in the repo behind `CLASSIFIER_HEAD=stem_recalib`, documented as a
  negative result rather than quietly used.
- **Max-pooling vs top-3-mean was a smaller effect than expected.** Aggregating
  chunk probabilities by mean-of-top-3 instead of max trades recall for
  precision and is roughly F1-neutral (BEATs 0.555 → 0.562; the CNN actually
  gets slightly worse, 0.513 → 0.509). It ships because precision matters more
  when a wrong instrument is visible on screen — not because it was a fix.
- **Two bugs survived a green test suite for two weeks** because the test mocked
  `soundfile.write`. Mocking an I/O boundary proves the surrounding logic, not
  that the plumbing works. There is now an unmocked round-trip test.
- **The chat eval harness found a hole in the grounding contract.** Generic stem
  words (`bass`, `vocals`, `drums`) were allow-listed unconditionally, so the
  agent could assert "there is a bass line" about a track whose bass stem is
  silent. Now gated on the presence flags.

## Limitations

- Trained on OpenMIC-2018: 20 instrument classes, weak labels, Western popular
  music. Anything outside that distribution is guesswork.
- Real-song F1 (0.562) is well below the OpenMIC number (0.8045). Clip-level
  benchmarks flatter you; the deployment distribution is harder.
- Tempo comes from librosa's beat tracker, which halves or doubles on
  rhythmically sparse material. Key detection needs Essentia and is skipped
  when it isn't installed.
- The chat agent is grounded, not omniscient — it can only report what the
  classifier detected, and the classifier misses things.
- The deployed demo analyses three precomputed tracks. Your own audio runs locally.

## Layout

```
src/                    the pipeline — preprocessing, separation, features, classifier
scripts/                fetch_models, eval_openmic, eval_real_songs
evals/                  chat-agent eval harness + an independent grounding checker
tests/                  pytest suite
reference-app/frontend  the deployed Next.js app (chat agent lives in lib/agent)
reference-app/backend   FastAPI service for local uploads
docs/research/          the write-ups behind every number above
docs/journal/           day-by-day development log
```

## Licence

MIT. Vendored BEATs code is Microsoft's, MIT-licensed — see
[`src/classification/beats/NOTICE.md`](src/classification/beats/NOTICE.md).
