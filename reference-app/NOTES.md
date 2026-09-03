# Decomposition reference app — study notes

**What this is:** a complete, working reference implementation of the final
product, built by Claude as a *teaching artifact*. Everything runs for real —
real Demucs, the real BEATs classifier from the research, a real local LLM.
Read it, run it, poke it, then **re-implement it in your own `src/`** — the
goal is that you could rebuild every file from understanding.

---

## 1. Run it locally

```bash
# terminal 1 — backend (uses the repo's existing .venv; deps: fastapi uvicorn python-multipart)
cd reference-app/backend
../../.venv/bin/uvicorn app:app --port 8000

# terminal 2 — frontend
cd reference-app/frontend
npm run dev          # → http://localhost:3000

# optional but recommended — local LLM for descriptions
ollama serve &       # if not already running
# (llama3.2:3b is already pulled; without Ollama the app falls back to a
#  deterministic template description — it never breaks)
```

Model weights load from `data/f1_research/` (BEATs checkpoint + trained
heads) — nothing was copied, see `backend/settings.py:MODEL_DIR`.

## 2. Architecture — the one diagram to internalize

```
Browser (Next.js on :3000)
  │  POST /api/analyze  (multipart upload)          ┐
  │  ← { job_id }                                   │ proxied by next.config
  │  GET /api/jobs/{id} every 2s ("polling")        │ rewrites to :8000
  │  GET /api/files/{id}/vocals.mp3 (stem audio, uploads only) ┘
  │  GET /examples/*  (precomputed demos — static, no backend)
  ▼
FastAPI (:8000)
  ├─ validation: magic bytes, size, duration        (app.py)
  ├─ abuse guards: per-IP cooldown, daily cap       (app.py)
  ├─ job store: dict + Queue + ONE worker thread    (app.py)
  ├─ pipeline: Demucs → BEATs head → BPM/key → LLM  (pipeline.py)
  ├─ classifier: frozen BEATs + MLP head            (models_util.py)
  └─ LLM: OpenAI-compatible client + grounding      (llm.py)
```

**Why async jobs + polling instead of one long HTTP request:** Demucs takes
minutes; browsers/proxies time out long requests, and a killed request wastes
the compute. So the POST returns instantly with an id and the client polls.
This is the universal ML-serving pattern — production systems swap the dict
for Redis/Postgres and the thread for worker processes, but the *shape* is
identical. (SSE/WebSockets are fancier alternatives to polling; polling is
the most robust and debuggable, which is why we start there.)

**Why ONE worker thread:** Demucs needs ~3 GB RAM. Two concurrent jobs would
OOM a free 16 GB host once models are loaded. Serializing jobs *is* the
capacity plan. Honest capacity > optimistic crash.

**Why models load at import time (module-level):** loading BEATs takes ~2 s.
Loading per-request would dominate latency; loading once per process is the
standard serving pattern ("warm worker").

## 3. File-by-file walkthrough (backend)

- **`settings.py`** — all config from env vars (12-factor pattern). The same
  code runs on your laptop and any host; only the environment differs.
- **`app.py`** — the web layer. Study, in order: `_sniff` (magic bytes —
  extensions lie; the first bytes of a file don't), the guards in
  `create_job` (cheapest checks first), `_worker` (the queue consumer),
  `_janitor` (TTL cleanup: uploads are ephemeral *by design* — public
  services must not hoard user audio). Note the path-traversal guard in
  `job_file` — never trust a filename that came from a URL.
- **`pipeline.py`** — the ML pipeline, mirroring your `src/main.py::analyze`
  but upgraded per the research: MP3 stems (10× smaller than WAV for the
  browser), progress callbacks (users need to know *why* they wait), presence
  via framed RMS (same idea as your `stem_presence.py`).
- **`models_util.py`** — the classifier. Key ideas: frozen BEATs + tiny head
  (F1 doc §8 — the champion, 0.8045), 5-second-window embedding protocol,
  **top-3-mean aggregation not max** (max-pooling measurably inflates false
  positives on full songs — F1 doc §6), stem-recalibrated head as deployment
  default (better real-song recall — trained on Demucs-processed clips so
  train matches deployment).
- **`llm.py`** — the swappable LLM layer. Three ideas worth stealing:
  (1) OpenAI-compatible API = one client for Ollama/Groq/Cerebras/anything;
  (2) hallucination control as an *executable contract* (validate → retry
  with feedback → deterministic fallback), never a hope;
  (3) graceful degradation: LLM down ≠ app down.

## 4. File-by-file walkthrough (frontend)

- **`next.config.ts`** — the `/api` proxy. Locally the browser talks to one
  origin (no CORS); in prod you point `BACKEND_ORIGIN` at the deployed API.
- **`app/page.tsx`** — landing. Server component shell; interactivity is in
  client components (`"use client"` only where needed — smaller bundles).
- **`components/UploadZone.tsx`** — upload UX. Note: backend errors (429 rate
  limit, 413 too big) surface as inline text — the server is the source of
  truth for rules; the client just displays them.
- **`app/track/[jobId]/page.tsx`** — the polling loop, the staged progress
  UI, and the example path (ids starting `ex_` read precomputed results).
- **`components/Console.tsx`** — the centerpiece. Four WaveSurfer instances,
  one controller: play/pause/seek fan out to all four; mute/solo is just
  per-stem volume. *Production upgrade path:* sample-locked sync via a single
  Web Audio `AudioContext` clock — noted, not needed for a demo.
- **Design system** (`globals.css`): "studio console" — near-black chassis,
  VU-meter amber, per-stem channel colors like a DAW, mono for data. A
  distinct point of view beats a generic component library for a portfolio.

## 5. Gotchas hit while building (so you don't re-hit them)

1. **kaldi fbank doesn't run on MPS** — BEATs' preprocessing runs on CPU,
   the transformer on GPU (`models_util._embed_10s_chunks` splits them).
2. **`soundfile` writes MP3 natively** (libsndfile ≥1.1) — no ffmpeg needed
   for stem encoding, but ffmpeg must exist for some librosa decodes.
3. **Magic bytes for m4a** live at offset 4 (`ftyp`), not 0.
4. **Word-boundary matching in the grounding validator** — "organic" is not
   "organ" (this exact bug happened in research).
5. **wavesurfer v7 `interaction` event** gives you the seek time — use it to
   fan seeks out to the other instances.
6. **Next 15: `params` is a Promise** in page components — unwrap with
   `use(params)`.
7. **Upload duration check needs a decode** — do it at upload time so bad
   files fail in seconds, not after queueing behind a 3-minute job.

## 6. Deploying it public + free (no card anywhere → no bill possible)

Two paths, both documented; do A first.

### Deployment — see [`DEPLOY.md`](../DEPLOY.md)

**The HF Space options that used to be described here are dead.** Docker Spaces
now require HF PRO ($9/month); free personal accounts get static Spaces and up
to two ZeroGPU Gradio Spaces only.

What ships instead: **one Vercel project**. The examples are static assets under
`frontend/public/examples/`, so the public site needs no server and never cold
starts, and the chat agent is a Next.js route handler (`app/api/chat/[jobId]`)
talking to Groq. Uploads run locally against this FastAPI backend — Demucs plus
a 345 MB model is more than any free serverless tier will give you.

`space/` is kept as a record of the abandoned approach, not a live path.

## 7. Map back to the research docs

- Classifier + thresholds + aggregation: F1 doc §6–8
  (`docs/research/2026-07-21-f1-improvement-research.md`)
- Architecture, hosting numbers, LLM decision: MVP doc §2–4, §7b, §11c
  (`docs/research/2026-07-22-mvp-webapp-gemini-plan.md`)
- Everything reusable (checkpoints, embeddings, eval scripts):
  `data/f1_research/`

## 8. Re-implementation ladder (suggested order for your own build)

1. `pipeline.py` equivalent in your `src/` (you already have most pieces).
2. Minimal FastAPI: analyze-sync endpoint first, then add the job queue.
3. Hardening (validation → rate limits → janitor) — one guard at a time.
4. Plain frontend page that polls and lists results as text.
5. WaveSurfer console (this is pure frontend fun).
6. LLM layer last — the contract/validator is the part worth writing
   carefully.
7. Deploy path A, then path B.

## 9. Verified end-to-end (screenshots)

Captured from the running app via headless Chromium (`reference-app/screenshots/`):
- `01_landing.png` — landing: hero, upload slot, example shelf
- `02b_console_waveforms.png` — example console: 4 stem waveforms, transport, meters, LLM writeup
- `03_processing.png` — live upload mid-Demucs: staged progress
- `04_result_live_upload.png` — a real song uploaded through the UI, fully analyzed (guitar 98%, 132.5 BPM, C# minor, llama3.2 description)

One real bug was found and fixed during this verification (see the comment in
`components/Console.tsx`): React StrictMode's mount→cleanup→mount cycle left
destroyed WaveSurfer instances in a ref, blocking waveform creation — the kind
of bug only an actual browser run surfaces. Lesson: always run the real thing.
