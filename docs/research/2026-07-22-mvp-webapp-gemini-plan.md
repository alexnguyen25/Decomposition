# Decomposition MVP: Web App + Gemini — Research & Implementation Plan

> **⚠️ SUPERSEDED IN PART (September 2, 2026).** §11c's "fully-free all-in-one
> Hugging Face Space" is no longer possible: **Docker Spaces now require HF PRO
> ($9/month)**. Free personal accounts get static Spaces and up to two ZeroGPU
> Gradio Spaces only. The site deploys to **Vercel Hobby** instead (free, no
> card, no cold start), serving precomputed analyses as static assets with the
> chat agent as a Next.js route handler on **Groq**. Uploads run locally. See
> [`DEPLOY.md`](../../DEPLOY.md). The §6 Gemini comparison also still describes
> the **E7/PANNs** head, not the shipped BEATs head, and would need rerunning
> before it goes in the README.
**Date:** July 22, 2026
**Role note:** All research/prototypes by Claude in a session scratchpad (synced to `data/f1_research/`); no `src/` changes, nothing committed. Alex implements everything.

> **Status: COMPLETE** (July 22, 2026). Headlines: $0/month split architecture validated with measured limits; HF Space demo built and core-verified; **classifier beats zero-shot Gemini 0.875 vs 0.598 masked macro-F1** on identical held-out clips; grounded Gemini app feature tested 6/6.

---

## 1. Executive summary

- **Architecture:** Next.js frontend on **Vercel** (your preference — validated for the frontend role only) + **Vercel Blob** for file storage + **Modal** for the heavy pipeline (T4 GPU, recurring $30/mo free credits). Estimated cost: **~$0/month**. A **Hugging Face Space** (Gradio, free) ships first as a standalone demo artifact — the ready-to-push folder already exists.
- **Why not Vercel alone:** measured and verified — Vercel functions cap at 4.5 MB request bodies (a song doesn't fit), 1 vCPU / 300 s / 2 GB (the pipeline needs ~3 GB and 2+ min), and the Hobby CPU-hour allowance would be consumed by ~50 songs even if it fit.
- **Pipeline requirements, measured on a real 4.5-min song:** 122 s total on an M-series Mac CPU (94% of it Demucs), peak RAM 3.0 GB. On free cloud CPU expect 4–15 min/song; on a Modal T4, well under a minute.
- **Gemini in-app feature: designed, tested, working.** Pipeline JSON → grounded 2–4 sentence track breakdown. 6/6 grounding-contract passes. Use **`gemini-flash-lite-latest`**: 0.8 s latency, ~10× cheaper than flash, equal quality on this constrained task.
- **Gemini benchmark: your classifier wins decisively.** E7 0.875 vs Gemini flash 0.598 / pro 0.569 masked macro-F1 on identical held-out clips; Gemini's failures concentrate on the confusable orchestral/wind classes. On real full songs Gemini is precision-biased (recall 0.28). Full protocol + tables in §6; total benchmark cost ≈ $1.

## 2. Recommended architecture

```
┌─────────────── Vercel (Hobby, $0) ───────────────┐
│  Next.js App Router frontend                      │
│  • upload page → @vercel/blob/client upload()     │
│  • thin API routes: create job / job status proxy │
└──────────┬────────────────────────────────────────┘
           │ POST /analyze {blobUrl}          ┌──────────────┐
           ▼                                  │ Vercel Blob  │
┌─────────────── Modal ($30/mo free) ───────┐ │ song.mp3     │
│  FastAPI ASGI app + spawned job containers │◄┤ stems/*.mp3  │
│  T4 GPU · torch + demucs + CNN14 + E7 head │ │ result.json  │
│  job: fetch song → Demucs → classify →     ├─► (public CDN) │
│       BPM/key → Gemini summary → write out │ └──────────────┘
└────────────────────────────────────────────┘
     job id → GET /status/{id} (poll 2s) or SSE
```

- **Job pattern:** client-upload to Blob (bypasses the 4.5 MB limit, has progress events) → `POST /analyze` returns `job_id` → **poll** `GET /status/{job_id}` every 2 s (SSE optional upgrade; WebSockets unnecessary and unsupported on Vercel functions anyway). Worker writes stems + JSON back to Blob; frontend streams stems from the CDN.
- **Why Modal over the alternatives** (full comparison §4): recurring free credits cover this workload ~15× over, GPU makes Demucs interactive (seconds vs minutes), async jobs and web endpoints are first-class, deploy is `modal deploy` from Python — no Dockerfile required.
- **HF Space as artifact #2:** the Gradio demo (already built, §5) is free to host, purpose-built for audio, and gives a resume-linkable URL even before the full web app exists. Ship it first — it de-risks everything else.

### Alternatives considered
| Option | Verdict | Why |
|---|---|---|
| Everything on Vercel | ✗ | 4.5 MB body cap, 1 vCPU/300 s/2 GB, CPU-hours exhausted by ~50 songs |
| Vercel + HF Space free CPU as backend | Workable fallback | $0 but 4–15 min/song and ~50 s cold starts; fine for demo, weak UX |
| Vercel + Fly.io | ✗ | no free tier (removed 2024), GPUs discontinued Aug 2026 |
| Vercel + Render free | ✗ | 512 MB RAM / 0.1 CPU can't hold the 3 GB pipeline |
| Vercel + Railway | Viable, worse | $5/mo, no GPU — strictly dominated by Modal here |
| All-in-one Gradio Space only | Good v1, not final | free, fast to ship, but less impressive than a real product frontend; ZeroGPU gives free GPU (2 Spaces/account) |

## 3. Measured pipeline profile (sizing evidence)

Real 4.5-min song (271 s), CPU-only, this machine:

| Stage | Time | Notes |
|---|---|---|
| Load audio | 0.9 s | |
| Demucs htdemucs separate | **115.3 s** | 94% of total; ~0.43× realtime on fast CPU |
| Resample + CNN14 embed + E7 head (27 chunks) | 5.9 s | classification is nearly free |
| **Total** | **122.4 s** | peak RSS **3.0 GB** |

Short clip (12.8 s): 9.7 s total, 1.9 GB peak. Implications: any host needs ≥4 GB RAM; Demucs dominates → GPU (Modal T4 / ZeroGPU) is the only lever that makes the demo feel interactive; model loads (~1 s Demucs, ~0.8 s CNN14) argue for a warm worker (load once per container, not per job).

## 4. Hosting comparison (verified July 2026)

| Host | Free tier | Fits 3 GB pipeline? | Cold start | ~50 songs/mo | Notes |
|---|---|---|---|---|---|
| **Modal** ⭐ | $30/mo recurring credits, no card | ✓ (any size, GPU) | seconds–1 min (optimized image loading) | **$0** (≈$2 usage) | T4 $0.59/hr, per-second billing; `modal deploy`; async jobs native |
| **HF Spaces CPU Basic** | free forever, no card | ✓ (2 vCPU/16 GB) | ~50 s+ after sleep | **$0** | 4–15 min/song; Gradio-native; sleeps after ~48 h idle |
| **HF ZeroGPU** | free ×2 Spaces (account >30 d) | ✓ (GPU slice) | fast once built | **$0** | Gradio SDK only; visitor GPU quotas (2–5 min/day free users) |
| Railway | $5 one-time trial, then $5/mo | ✓ (4 GB) | fast | ~$5/mo | no GPU |
| Fly.io | none (removed) | ✓ | seconds + model load | ~$1–3 usage | card required; GPUs discontinued |
| Render free | free | ✗ 512 MB | ~1 min | — | not viable |
| Vercel functions | Hobby | ✗ | — | — | frontend + Blob + thin API only |

Vercel specifics that shape the design: 4.5 MB function body limit (→ client Blob uploads), 300 s max duration, SSE ok / WebSockets not, Blob free within Hobby limits with hard stops (no surprise bills).

## 5. HF Space prototype — built and core-verified

Ready-to-push folder in `data/f1_research/space/`: `app.py` (Gradio UI: upload → 4 stem players + instrument list + BPM/key + JSON), `requirements.txt`, `README.md` with exact deploy steps (create Space → copy files + `Cnn14_mAP=0.431.pth` + `ckpt_E7_panns_head.pt` renamed `e7_head.pt` + `class-map.json` → `git lfs track "*.pth" "*.pt"` → push).

The full `analyze()` core ran locally end-to-end on the known test song: **BPM 132.5, key C# minor, guitar 0.984 — and zero false positives** (the day-17 baseline produced 13 predictions with 10 FPs on this exact song). E7 + top-3-mean is a dramatically more precise real-song configuration; the trade-off is missing this song's quieter piano/synth. Per-class threshold recalibration on stem-domain data (F1 doc §7 step 6) is the tuning knob.

Design decisions encoded in the app: models load once at module import (warm worker); top-3-mean aggregation (measured better than max, F1 doc §6); dead-weight classes excluded; `gr.Progress` stages so the slow Demucs step shows progress; essentia optional (key falls back to "n/a" — its Spaces build is flaky).

## 6. Gemini benchmark: classifier vs frontier model

**Methodology** (designed to be immune to "you tuned on the test set"):
- Prompt engineering on **15 train-partition clips only**, 3 variants: P1 plain, P2 + "only if confident", P3 + expert framing, class-disambiguation notes (e.g. mallet_percussion = xylophone/marimba/vibraphone), and a separate `uncertain` bucket excluded from scoring. Dev masked macro-F1: P1 0.178 < P2 0.218 < **P3 0.229** → P3 wins; conservatism and label disambiguation both help.
- Final: 100 stratified test clips (seed 42, ≥1 confirmed positive), `gemini-flash-latest` on all 100, `gemini-pro-latest` on 50, JSON response mode, audio as inline base64. **E7 and E4 scored on the identical clips with the identical masked metric.**
- Plus the 14 real Jamendo tracks (tag recall / extras-per-track, same protocol as the classifier eval).

**Results — 10-second OpenMIC clips (identical 100 clips, identical masked metric):**

| System | Masked macro-F1 |
|---|---|
| **E7 (PANNs + head, tuned thresholds)** | **0.875** |
| E7 @0.5 | 0.859 |
| E4 (from-scratch CNN) @0.5 | 0.793 |
| Gemini flash-latest (P3 prompt) | 0.598 |
| Gemini pro-latest (50-clip subset) | 0.569 |

(Note: E7 scores higher here than on the full 5085-clip test set (0.79) because this stratified positive-rich subset is easier — which is exactly why all systems are scored on the *same* clips.)

**Where the gap comes from (per-class, E7 vs flash):** Gemini collapses on the acoustically confusable orchestral/wind classes — cello 0.00, mandolin 0.00, clarinet 0.00, trumpet 0.20, accordion 0.33 — while matching the classifier on distinctive timbres (synthesizer 1.00, voice 1.00, organ 0.67). The specialized model's entire advantage is concentrated precisely where the F1 research showed generic models struggle. Honest caveats for the README table: Gemini is zero-shot with no threshold tuning, and a different prompt could shift its numbers somewhat — state the protocol (prompt, model version, date) next to the table.

**Real full-length songs (14 Jamendo tracks):** Gemini tag-recall 0.280 with 1.14 extras/track — extremely conservative next to the pipeline's E4/max (0.890 recall, 6.79 extras) and E7/max (0.622, 3.29). On long mixed audio it finds only the most obvious instruments but rarely invents. Fair summary: *the frontier model is precision-biased and misses most instrumentation; the specialized pipeline hears far more at a manageable false-positive cost.*

**Cost/latency (measured):** flash ≈ 3.6 s/clip; 100 clips ≈ 44k prompt + 47k output/thinking tokens ≈ **$0.13 per 100 clips** — the full benchmark costs under a dollar to reproduce. Reruns are free-ish: predictions cache to disk (`gemini_cache_*.json`), so a crash never re-pays for completed calls.

**Resume phrasing this supports:** "Custom classifier (macro-F1 0.875) outperformed zero-shot Gemini (0.598) by 0.28 macro-F1 on a held-out OpenMIC-2018 subset under an identical masked-metric protocol, with the largest gains on acoustically confusable orchestral instruments."

## 7. In-app Gemini feature — designed, tested, 6/6 grounding passes

**Feature:** after the pipeline finishes, Gemini turns the result JSON into a 2–4 sentence listener-facing "track breakdown."

**Grounding contract (tested, this is the important part):**
- Prompt embeds the JSON and forbids naming instruments not in it; confidence bands map to hedging language (≥0.90 plain, 0.70–0.89 "clear", 0.50–0.69 "hints of"), numbers never shown.
- Response is JSON: `{"summary", "mentioned_instruments"}`; the app **programmatically validates** `mentioned_instruments ⊆ (instruments ∪ presence-derived stems)` and rejects/retries on violation — hallucination becomes a testable contract, not a hope.
- Design subtlety found by testing: the contract must whitelist presence-derived stems (drums/bass/vocals) — the first version flagged legitimate mentions of them as hallucinations.

**Model choice, measured:** `gemini-flash-lite-latest` — 0.8 s vs flash's ~6 s (thinking overhead), ~10× cheaper, equal quality across 3 diverse cases. Per-song cost: fractions of a cent. Implementation detail: call it from the Modal worker (key stays server-side), write the summary into `result.json`.

### 7b. Upgrade: full song *description*, not just an instrument summary (tested)

Second iteration, tested on 3 real Jamendo tracks with real E7 pipeline output — richer schema: `{blurb, genre, moods[], energy, tempo_feel, era_production, mentioned_instruments[]}`.

Two grounding variants compared (× flash and flash-lite = 12 runs):
- **A. text-only** (JSON in, no audio): genre/mood are guesses from instrumentation+BPM. Failure mode found: when the classifier under-reports (one track's JSON contained only `piano`), text-only guessed "electronic" for what is audibly a classical ensemble piece.
- **B. multimodal (winner):** a 60-second mix excerpt goes in *alongside* the JSON — Gemini listens for genre/mood/energy/production, but instrument *naming* stays locked to the JSON ("describe anything else by texture/role, never by name"). It produced more specific, committed genres ("Polka" vs "Folk-jazz fusion") and correctly heard the classical character text-only missed.

**Grounding contract: 12/12 passes** across all variants/models. Two validator subtleties worth keeping (both found by testing): whitelist presence-derived stem mentions (drums/bass/vocals), and match class names on **word boundaries** — a naive substring check flags "organic production" as a leaked "organ."

**Ship:** multimodal on `gemini-flash-lite-latest` — 2.0–2.7 s, ~1.5k audio tokens per call (fractions of a cent). Locked prompt + schema + validator: `data/f1_research/gemini_description.py`; all 12 raw responses in `gemini_description_results.json`.

## 8. API contracts

```
POST /api/upload-url            (Vercel route)
  → { uploadUrl, blobUrl }      client uploads directly to Blob

POST /api/analyze               (Vercel route → proxies Modal)
  body: { blobUrl }
  → { jobId }
  409 if file >15 MB or >6 min; 400 on non-audio (sniff header server-side)

GET /api/jobs/{jobId}           (Vercel route → proxies Modal status)
  → { status: "queued"|"separating"|"classifying"|"summarizing"|"done"|"error",
      progress: 0-1, error?: string, result?: Result }

Result JSON (extends the current main.py shape — one schema everywhere):
{
  "bpm": 132.5,
  "key": "C# minor",
  "stems": { "vocals": "<blobUrl>", "drums": "...", "bass": "...", "other": "..." },
  "presence": { "vocals": true, "drums": true, "bass": true },
  "instruments": [ { "name": "guitar", "confidence": 0.984 } ],
  "summary": "Set in C# minor with a lively tempo...",   // Gemini, grounded
  "timings": { "separate_s": 21.3, "classify_s": 2.1, "total_s": 26.0 }
}
```

Modal side: one `@app.function(gpu="T4")` job + a small FastAPI endpoint pair (`spawn` → job id, `status` → poll). Progress updates via a Modal Dict keyed by job id.

## 9. Page-level UI spec (Next.js, 3 pages)

1. **Landing / upload** — hero line, drag-drop upload with client-side type/size check + progress bar, **3 preloaded example songs** (CC-licensed, results pre-cached in Blob → instant one-click demo for recruiters), link to GitHub + research docs.
2. **Analysis view** (`/track/[jobId]`) — progress stages while running (named steps, not a bare spinner: "Separating stems… ~30 s"); on completion: 4 stem rows (wavesurfer.js v7 waveform + play/mute/solo, one shared transport syncing all four — the official multitrack plugin is commercial, sync 4 instances manually), instrument list as confidence bars, BPM/key chips, Gemini summary paragraph, collapsible raw JSON, "download stems" links.
3. **About / research** — the 0.650→0.794 story, benchmark-vs-Gemini table, architecture diagram, links to `docs/research/`. This page is *for interviewers*; it's the moat.

## 10. Security notes

- **Gemini key:** server-side only (Modal secret: `modal secret create gemini GEMINI_API_KEY=...`); never in the browser, never in git (history already audited clean — keep it that way).
- **Uploads:** enforce size (≤15 MB) and duration (≤6 min) server-side; sniff magic bytes, don't trust extensions; never shell out with user filenames (use Python APIs, no `os.system`).
- **Blob:** uploaded songs are publicly-URL'd — random URLs are unguessable but not private; state this in the UI ("don't upload private audio"), and set a TTL cleanup job (delete blobs >7 days) to bound storage and copyright exposure.
- **Abuse:** rate-limit `POST /analyze` per IP (Vercel middleware) — each job costs GPU seconds and Gemini tokens.
- **Modal endpoint:** require a shared bearer token between Vercel routes and Modal so the compute endpoint isn't publicly drive-able.

## 11. Build-order ladder (independently shippable)

1. **Ship the HF Space** (folder is ready) → live demo URL exists for the resume *this week*. Add 2–3 cached examples.
2. **Implement the F1-doc Section 7 ladder in `src/`** (E7 head as the production classifier, top-3-mean, thresholds) so the repo's own pipeline matches what the demo serves.
3. **Modal worker:** port `main.py`'s analyze into a Modal function (T4), models in a Volume; verify one song end-to-end from `modal run`.
4. **Next.js frontend on Vercel:** upload → poll → results page against the Modal API (mock first, then real).
5. **Wire Blob storage + preloaded examples + rate limiting.**
6. **Add the Gemini summary** to the worker (grounded contract from §7, flash-lite).
7. **Run the Gemini benchmark yourself** (script: `gemini_bench.py`) and publish the table in README + About page.
8. **Portfolio polish pass** (checklist §12): README hero with GIF + metrics, architecture diagram, CI badge, license, pinned repo.

## 11c. Fully-open, $0, public deployment — no Gemini dependency (July 22, third pass)

New priority from Alex: a **real public website** (not a notebook/Gradio) that anyone can use, **fully free with no bill possible and no proprietary-LLM dependency** — open weights, swappable.

**"Local" vs "public" — the constraint to design around:** Ollama on the MacBook is dev-only; a public site can't depend on a personal machine as an always-on server. The resolution: an **OpenAI-compatible LLM abstraction** (one client, swap `base_url`+model via env). Ollama exposes `/v1/chat/completions`; Groq, Cerebras, OpenRouter, HF, and Gemini all do too. Local dev → Ollama; public prod → a free *hosted open* model.

**LLM decision (measured):**
- Local **Llama 3.2 3B** via Ollama on Alex's M-series Mac: **3/3 grounding-contract passes, 3.0 s/call** — but left the `genre` field as the literal placeholder in 2/3 cases (small models pass grounding yet under-fill structured fields; matches published small-LLM structured-output benchmarks). On a free HF CPU backend expect ~10–20 s/call and RAM contention with Demucs.
- **Recommended prod default: a free *hosted open-weight* model** behind the same abstraction — **Cerebras (1M tokens/day, no card)** or **Groq (~14k req/day, no card, very fast)**, both open-weight (Llama/Gemma), both hard-stop rather than bill. Gives ~8B-class quality (better field-filling than a self-hosted 3B) at $0 with no key-spend/bill risk.
- **Self-hosted 1–3B on the backend** = offline fallback only (works, slower, weaker fields).
- **Drop the multimodal/audio LLM path:** the only thing that required Gemini specifically. The **BEATs classifier already *is* the ears**, so feed its JSON to a text LLM. Open audio models (Qwen2-Audio 7B) are too heavy for free compute anyway. This removes the last Gemini dependency entirely.
- Keep the grounding contract + word-boundary validator from §7b regardless of provider.

**Fully-free public stack (all verified mid-2026, no card / no bill possible):**
- **Frontend:** Cloudflare Pages (Next.js) — a real website, unlimited bandwidth, custom domain, keeps serving at limits (no overage bill). Vercel Hobby also works but is non-commercial and pauses at limits.
- **Backend compute:** HF Space **CPU Basic (2 vCPU/16 GB, 2000 free CPU-hrs/mo, no card)** running **FastAPI** (not Gradio, so the frontend stays a real site) — fits Demucs (~3 GB) + BEATs (~0.4 GB) + optional small LLM. Sleeps after 48 h idle, ~30–60 s cold wake. Alternative always-on option: **Oracle Cloud Always Free ARM (2 OCPU/12 GB)** — no sleep, but card required at signup and CPU-only.
- **LLM:** free hosted open model (Cerebras/Groq) via the abstraction.
- **$0 guarantee:** HF free tier + Cloudflare Pages + no-card LLM keys = nothing has a card attached, so every limit is a hard stop, never a charge. Add per-IP rate limiting + a global daily cap + cached demo-song results.
- **Speed catch (honest):** free CPU = 1–5 min/song (Demucs dominates). Show a progress UI and preloaded one-click examples. HF **ZeroGPU** (free, 2 Spaces/account) makes it seconds but caps anonymous visitors at ~2–5 GPU-min/day total — good for low traffic, throttles under a crowd.

**Net:** fully open-weight, free, public, real website, and *simpler* than the Gemini design (no key to protect, no bill to cap). Artifacts from this pass: `data/f1_research/local_llm_test.py`, `local_llm_results.json`. This supersedes §11b's Gemini-only LLM step: build the swappable LLM layer (Ollama dev / Cerebras|Groq prod), keep Gemini only as one optional provider.

## 11b. Build-ladder revision (July 22, second research pass)

The follow-up experiments (F1 doc §8) change three things:

1. **The production classifier is now frozen BEATs embeddings + MLP head — 0.8045 tuned macro-F1** (beats PANNs E7 0.7915, the CNN14 fine-tune, and every ensemble). Ladder steps 2–3: extract BEATs embeddings (code + 352 MB checkpoint in `data/f1_research/beats/`, cached embeddings in `beats_split01_*.npz`), train the head, and ship BEATs in the Space/Modal worker instead of CNN14. Same 16 kHz mono input path; two 5 s windows per 10 s chunk, mean-pooled.
2. **For the real-song pipeline, fine-tune that head on stem-domain data** (recipe proven on E7: 300 Demucs-processed train clips ×3 oversampling, 4 epochs, lr 2e-4) — it lifted real-song tag recall 0.585 → 0.659. The 300 cached stems are in `data/f1_research/stem_audio_cache/`. Skip stem-domain *threshold* tuning (overfits at this calibration size).
3. **Do not bother fine-tuning backbones end-to-end** — measured tie with frozen embeddings at 300× the training cost (F1 doc §8, E9).
4. The Gemini in-app feature ships as the **multimodal description** (§7b): 60 s excerpt + JSON on `gemini-flash-lite-latest`, 12/12 grounding contract.

Benchmark table note: the §6 Gemini comparison used E7 (0.875 on the 100-clip subset); with BEATs the gap vs Gemini (0.598) widens further — rerun `gemini_bench.py`'s classifier arm with the BEATs head when implementing if you want the updated number in the README.

## 12. Resume/portfolio checklist (research-backed, ordered by impact)

Recruiters skim ~7 s on resumes, ~90 s on GitHub; a live URL beats everything (nobody clones during screening). Full findings + sources in the research notes; the executable list:

1. Live demo URL at the top of README, resume, repo About field, LinkedIn
2. 2–3 preloaded songs with cached stems → one-click, <10 s to first result
3. README hero above the fold: one-sentence pitch, 15–30 s GIF, metrics table (0.650→0.794 + vs-Gemini), architecture diagram
4. "Research log" README section linking `docs/research/` — include the failures (SpecAugment hurt; max-pooling FP explosion); interviewers probe trade-offs and failures
5. Resume bullets = baseline→final + method + dataset: *"Improved instrument-classification macro-F1 0.650→0.794 on OpenMIC-2018 by benchmarking 8 approaches (scratch CNNs → pretrained audio embeddings); benchmarked against Google Gemini on a held-out test set"* — claim only measured facts, per-instrument phrasing for the Gemini comparison
6. One-command quickstart tested in a clean venv; tests + GitHub Actions CI badge; MIT license; `.gitignore` hygiene (already clean — keep the key out)
7. Move stray logic out of notebooks; keep one narrative notebook max; clean commit messages going forward
8. "Limitations & future work" section — honesty reads as maturity
9. Uptime-check the demo before every application season (dead demo < no demo)
10. Prepare 3 interview stories: a trade-off (PANNs vs scratch CNN), a failure (SpecAugment/max-pooling), a measurement decision (masked macro-F1, val-split thresholds)

## 13. Reproducibility

Everything from this phase lives in `data/f1_research/` (synced from the session scratchpad): `gemini_bench.py` (+ `gemini_bench_*.json`), `gemini_feature.py` (+ results), `profile_pipeline.py` (+ `pipeline_profile.json`), `space/` (deployable Space), plus everything from the F1 phase. Gemini key in `.env` (gitignored). Models that work on this key: `gemini-flash-latest`, `gemini-flash-lite-latest`, `gemini-pro-latest` (pinned names like `gemini-2.5-flash` 404 for new projects — use the aliases).
