# Feasibility Report: Lyrics Transcription + Chord/Structure Detection on Demucs Stems

**Date:** 2026-07-31
**Context:** The app already separates songs into 4 stems with Demucs (htdemucs): vocals, drums, bass, other. This report assesses two candidate features: (A) timestamped lyrics transcription on the vocal stem, and (B) chord + song-structure detection. Research only — nothing was installed or implemented.

---

## Executive summary

| Feature | Verdict | One-line rationale |
|---|---|---|
| **A. Lyrics transcription on separated vocals** | **GO** (with VAD gating as a hard requirement) | Whisper-family ASR on music is well-studied; ~20–25% WER on English pop is the realistic open-source ceiling. We already produce the vocal stem, so the marginal cost is one ASR pass. Hallucination on instrumental sections is the #1 risk and has known mitigations. |
| **B1. Chord detection** | **GO, but with modest quality expectations** | The library landscape is mostly dead or research-grade. Best pragmatic options: BTC pretrained weights (research code, PyTorch) or Chordino via `chord-extractor` (simple, classic quality). Expect ~75–80% triad accuracy at best; stem-assisted preprocessing gives only marginal, mixed gains. |
| **B2. Structure (verse/chorus) + beats/tempo** | **GO — strongest of the three** | `allin1` does structure + beats + downbeats + BPM in one model, uses Demucs stems internally (conceptual match with our pipeline), has an actively maintained Apple-Silicon port (`all-in-one-mlx`, Mar 2026). SongFormer (ISMIR-adjacent, Oct 2025) is the newer SOTA with released code. |
| **B4. Swap librosa key/BPM?** | **NO-GO for now** | librosa has no built-in key detector anyway (any key output we have is custom code); essentia is actively maintained with good key profiles but is **AGPL-3.0**, which is a licensing landmine for an app. Keep librosa; if we adopt allin1 we get beats/tempo upgraded for free. |

---

## Question A — Lyrics transcription on separated vocals

### A1. What the literature says: separated vocals vs. full mix

The most directly relevant study is **"Exploiting Music Source Separation for Automatic Lyrics Transcription with Whisper"** (Cífka et al., ICME 2025 workshop, [arXiv:2506.15514](https://arxiv.org/abs/2506.15514), [HTML](https://arxiv.org/html/2506.15514v1)). Setup: Whisper **large-v2**, Hybrid Demucs variants (`mdx` 7.97 dB vocal SDR, `mdx_extra` 8.76 dB). Key numbers:

| Condition | Jam-ALT WER | MUSDB-ALT WER |
|---|---|---|
| Original mix (native long-form decoding) | 23.02% | 25.82% |
| Separated vocals, `mdx_extra` | 22.87% (negligible change) | 21.90% |
| Ground-truth vocal stems | — | 17.51% |
| Original mix + their **RMS-VAD** | **20.35%** (open-source SOTA on Jam-ALT) | 22.72% |

Takeaways relevant to us:

- **Separation helps when the accompaniment is loud/dense** (MUSDB-style mixes: 25.8% → 21.9%), and is **roughly neutral on typical commercial mixes** (Jam-ALT: 23.0% → 22.9%). True clean stems (17.5%) show the headroom that better separation could unlock.
- **VAD-style gating helped more than separation did** on Jam-ALT (−2.7 pts WER) and also reduced hallucination-derived insertions across all conditions.
- Caveats the paper reports: separation artifacts **can trigger hallucinations**; separation does **not** fix deletions of backing vocals and non-lexical vocables ("la-da-da"); on separated vocals Whisper more often outputs the **wrong language**; and large-v3 is not necessarily better than large-v2 on lyrics.
- Per-language WER (mix + RMS-VAD): German 16.1%, Spanish 14.3%, English 24.7%, French 24.4% — i.e., **~20–25% WER is the realistic expectation for English pop with open-source models**, not speech-like 5%.

Benchmarks/datasets to know:

- **Jam-ALT** ([site](https://audioshake.github.io/jam-alt/), [arXiv:2311.13987](https://arxiv.org/abs/2311.13987)) — 79 full songs, EN/FR/ES/DE, readability-aware ALT benchmark (ISMIR 2024); line-level timings added by the ICME 2025 paper above. The maintainers (AudioShake) report their commercial system cuts WER ~57% vs. Whisper v2 ([blog](https://www.audioshake.ai/post/new-benchmark-for-higher-quality-lyrics-transcription-from-audioshake-research)) — a useful ceiling reference.
- **MUSDB-ALT** — lyrics for the MUSDB18 test set (same paper), useful because MUSDB is exactly the domain our Demucs model was trained on.
- **Music.AI's lyric transcription benchmark** ([blog](https://music.ai/blog/research/lyric-transcription-benchmark/)) compares commercial providers on CER/WER — confirms lyrics WER is far above speech WER for everyone.
- Community anecdote: whisper.cpp model comparison on Queen's "Don't Stop Me Now" ([discussion #3074](https://github.com/ggml-org/whisper.cpp/discussions/3074)) — all large variants mangle or drop non-lexical vocals; turbo replaced the "la-da-da" outro with "Thank you."

**Bottom line for A1:** transcribing our **htdemucs vocal stem** instead of the mix is a reasonable default (we get it for free), but the literature says the bigger win is **energy/VAD gating using the stem**, and results should be validated locally because separation sometimes hurts (artifacts → hallucinations, wrong-language flips).

### A2. Best practical setup in 2026

**On Mac (M-series):**

- **mlx-whisper** is the standard Metal-accelerated backend; community benchmarks put MLX implementations well ahead of CPU-bound alternatives on Apple Silicon (one base-model benchmark: openai-whisper 12.3× realtime, faster-whisper 17.3×, mlx-whisper 29.7×, lightning-whisper-mlx 36.4× — [whisper-bench](https://github.com/naveedn/whisper-bench), [mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest)). Note **faster-whisper/CTranslate2 has no Metal backend** — it runs CPU-only on Mac, so MLX wins there ([2026 overview](https://bitneuronal.com/p/transcripcion-local-con-whisper-en-2026-faster-whisper-mlx-para-apple-silicon-y-diarizacion)).
- For **word-level timestamps**: Whisper's native word timestamps (DTW on cross-attention) work in mlx-whisper; for better word timing, **WhisperX**-style forced alignment (wav2vec2) is the standard. WhisperX itself is faster-whisper-based ([m-bain/whisperX](https://github.com/m-bain/whisperX), actively maintained, v3.7.x releases in Jan 2026, VAD preprocessing + word alignment + batching); an MLX-backend fork exists ([whispermlx](https://github.com/KalebJS/whispermlx)). Caveat: wav2vec2 alignment models are trained on **speech**, so alignment quality on singing is an unknown to measure (see Unknowns).
- **whisper.cpp** is the other viable Mac path (Metal, quantized models), good for shipping a lightweight binary.

**On a cloud T4:**

- **faster-whisper** (v1.1.x, [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)) is the default: ~4× faster than openai/whisper at equal accuracy, large-v3 fits in <8 GB VRAM with fp16/int8, built-in **Silero VAD filter** and `word_timestamps=True`. Batched mode gives large additional speedups. A T4 (16 GB) handles large-v3 comfortably.
- **WhisperX** on top of faster-whisper if we want the wav2vec2-aligned word timestamps + diarization stack.

**Model choice:**

- **large-v3**: best raw multilingual accuracy but documented **higher hallucination propensity** than v2 ([AssemblyAI comparison](https://www.assemblyai.com/blog/comparing-universal-2-and-openai-whisper)); the ALT paper found large-v3 not better than large-v2 on lyrics. Worth A/B-ing v2 vs v3 locally.
- **large-v3-turbo**: 809 M params, 4 decoder layers, ~6× faster, within 1–2% of large-v3 **on speech** ([HF card](https://huggingface.co/openai/whisper-large-v3-turbo)); on music it showed the worst vocable handling in the whisper.cpp shootout. Good candidate for a "fast mode," not for the quality path.
- **distil-whisper distil-large-v3.5**: English-only; word timestamps exist but alignment heads were **not retrained during distillation** (only segment-level timestamps distilled), so word timing is second-class ([HF discussion](https://huggingface.co/distil-whisper/distil-large-v3.5/discussions/6), [model card](https://huggingface.co/distil-whisper/distil-large-v3.5)). Skip for timestamped lyrics.
- **NVIDIA Parakeet-TDT 0.6B v2/v3 and Canary-1B-v2** (NeMo): native punctuation/capitalization and **word-level timestamps**; the Parakeet model card explicitly claims robust performance on **song lyrics transcription** ([HF card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [paper](https://arxiv.org/pdf/2509.14128)); extremely fast/cheap in batch settings ([Modal writeup](https://modal.com/blog/fast-cheap-batch-transcription)). Cons: NeMo dependency stack is heavy; no MLX/Metal path (CPU-only on Mac); v2 is English-only (v3/Canary multilingual). **Worth including in the local bake-off on the T4 side** — it's the only open model family that advertises lyrics robustness.

**Practical recommendation:** Mac dev path = mlx-whisper large-v3 (or v2) on the vocal stem with VAD gating; cloud path = faster-whisper large-v3 (int8/fp16) or WhisperX, with Parakeet-TDT as a challenger. One pipeline design, two backends.

### A3. Known failure modes and standard mitigations

Failure modes (all documented above and in community threads):

1. **Hallucination on instrumental/silent sections** — classic outputs: "Thank you for watching", repeated loops, YouTube-ese; Whisper is trained on captioned web audio ([openai/whisper #679](https://github.com/openai/whisper/discussions/679), [Calm-Whisper, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.pdf)).
2. **Repetition loops** on long instrumental stretches (decoder conditioning runaway).
3. **Deleted non-lexical vocables and backing vocals** — not fixed by separation (ICME 2025 paper).
4. **Wrong-language output**, more frequent on separated vocals (ICME 2025 paper).
5. **Rap / fast dense delivery** — high WER; no specific open benchmark found, flag for local testing.
6. **Heavy FX (autotune, distortion, reverb tails)** — degrades both ASR and the separation itself; anecdotal, measure locally.

Standard mitigations, in rough order of value:

- **VAD/energy gating**: only transcribe regions where the *vocal stem* has energy. The ICME 2025 paper's RMS-VAD (simple RMS thresholding on the separated vocal) beat native decoding and cut insertions — this is tailor-made for us since we already have the stem. Silero VAD (built into faster-whisper/WhisperX) is the off-the-shelf alternative, but it's trained on **speech**: it both misses sung vocals sometimes and passes noise (~61% utterance accuracy on non-speech noise per [faster-whisper #843](https://github.com/SYSTRAN/faster-whisper/issues/843)); a two-tier "vocal islands" design (VAD gates Whisper output) is a known community pattern ([example writeup](https://note.com/airin326/n/n16c3eaa04344?hl=en)).
- **Decoding hygiene**: `condition_on_previous_text=False`, `no_speech_threshold` + `logprob_threshold` tuning, compression-ratio filter to drop loop-y segments, temperature fallback off for determinism.
- **Post-filtering**: drop segments whose time span has near-zero vocal-stem RMS (we can compute this exactly); blacklist known hallucination strings.
- **Research-grade**: Calm-Whisper (attention-head calming) exists but is not a packaged tool; not worth adopting.

---

## Question B — Chord + structure detection

### B1. Chord-recognition libraries: 2026 health check

| Tool | Status 2026 | Python compat | Quality reputation | Notes |
|---|---|---|---|---|
| **madmom** ([PyPI](https://pypi.org/project/madmom/), [GitHub](https://github.com/CPJKU/madmom)) | **Effectively dead on PyPI**: last release 0.16.1, Nov 2018; classifiers stop at Python 3.7; downstream projects report it blocks Python ≥3.10 ([beat_this #9](https://github.com/CPJKU/beat_this/issues/9)) | Git-install from `main` works on newer Pythons with pinned numpy/cython; fragile | CNN+CRF chord model (`CNNChordFeatureProcessor` + `CRFChordRecognitionProcessor`) was long the best packaged option; maj/min vocabulary only | allin1 depends on it (installed from git). **Verify model license before shipping** — madmom's trained models have historically carried a non-commercial CC license separate from the BSD code. |
| **autochord** ([GitHub](https://github.com/cjbayron/autochord), [PyPI](https://pypi.org/project/autochord)) | Dormant (ISMIR 2021 LBD; last release 0.1.3) | Old TF dependency; expect friction on modern stacks | Self-reported 67.3% test accuracy, **25 classes (maj/min + N) only** | Not recommended. |
| **crema** ([GitHub](https://github.com/bmcfee/crema), [docs](https://crema.readthedocs.io/)) | Dormant (~97 stars, low activity; bmcfee project) | `python>=3.6`, `tensorflow>=2.0`, `keras>=2.6` — **Keras 3 / modern TF era makes this risky**; needs a pinned env | Solid ISMIR-era chord model with structured vocabulary (root/pitch-class outputs), JAMS output | Usable if sandboxed in its own venv; don't build the app on it. |
| **essentia** ([PyPI](https://pypi.org/project/essentia/), [GitHub](https://github.com/MTG/essentia)) | **Actively maintained** — 2.1b6.dev1438 released May 2026, wheels incl. macOS arm64 and manylinux | Modern CPython (wheels up to 3.14) | `ChordsDetection` is template-matching on HPCP chroma — basic quality, maj/min; well below neural models | **AGPL-3.0** — significant license constraint for an app. |
| **chord-extractor** ([GitHub](https://github.com/ohollo/chord-extractor), [PyPI](https://pypi.org/project/chord-extractor)) | Maintained-ish; author still using it in 2025 ([lmd_chords dataset](https://huggingface.co/datasets/ohollo/lmd_chords)) | Bundles compiled **Chordino** (NNLS chroma VAMP plugin) for Linux x64; macOS needs manual VAMP setup | Chordino = classic pre-deep-learning quality; recognizable but dated | Easiest "it just works" path on Linux; batch/multiprocessing built in. |
| **BTC** ([GitHub jayg996/BTC-ISMIR19](https://github.com/jayg996/BTC-ISMIR19), [paper](https://arxiv.org/abs/1907.02698)) | Research code, unmaintained but **PyTorch + pretrained weights available**; still the reference open model — the APSIPA 2025 stem paper builds on it | PyTorch, so portable; needs light modernization (no packaging) | ~75.5% triads WCSR on Isophonics/RW/uspop2002 (APSIPA 2025 reproduction); large-vocab (170 chords) checkpoint exists | **Best quality-per-effort open option** if we accept vendoring research code. Weights license: check repo (research release). |
| **consonance-ACE** ([GitHub andreamust/consonance-ACE](https://github.com/andreamust/consonance-ACE), [ISMIR 2025 paper](https://arxiv.org/abs/2509.01588)) | **New (ISMIR 2025)**, code + trained conformer released; outputs .lab files | Modern PyTorch | Conformer with decomposed root/bass/note training; explicitly predicts bass → better inversions | The most modern open chord model; maturity unknown — evaluate. |
| Omnizart, Harmony-Transformer, etc. | Dead or research-only | — | — | Not worth pursuing. |

**Recommendation:** prototype with **BTC pretrained** (quality) and **chord-extractor/Chordino** (simplicity) side by side; keep **consonance-ACE** as the stretch option. Avoid building on madmom/crema/autochord.

### B2. Structure segmentation tools

| Tool | Status 2026 | What it outputs | Notes |
|---|---|---|---|
| **allin1** ([GitHub mir-aidj/all-in-one](https://github.com/mir-aidj/all-in-one), [PyPI 1.1.0, Oct 2023](https://pypi.org/project/allin1/), [paper ICASSP 2024](https://arxiv.org/abs/2307.16425)) | Semi-dormant: last push May 2024, 18 open issues, not archived; still the community default | **Tempo (BPM), beats, downbeats, segment boundaries + functional labels** (intro/verse/chorus/bridge/outro), JSON, plus frame-level activations/embeddings | **Confirmed: it runs htdemucs internally** ("demixed audio" is the core of the model) — conceptually perfect for our pipeline, though its API does its own demixing (double-Demucs unless we patch/cache). Pain points: **NATTEN** (source build on some platforms), **madmom from git** (Python-version fragility), Python ≥3.8. Trained on Harmonix Set; SOTA-at-publication on beats/downbeats/structure. |
| **all-in-one-mlx** ([PyPI 1.0.5, Mar 2026](https://libraries.io/pypi/all-in-one-mlx)) | **Actively maintained fork** (Sam Small / ssmall256) | Same outputs, end-to-end on Apple Silicon MLX; claims ~12.6× faster than upstream on M4 Max; Python ≥3.10, macOS 14+ | Solves the Mac story for allin1. Cloud/T4 still uses upstream or `all-in-one-fix` ([PyPI](https://pypi.org/project/all-in-one-fix/), community dependency-fix fork). |
| **SongFormer** ([GitHub ASLP-lab/SongFormer](https://github.com/ASLP-lab/SongFormer), [arXiv:2510.02797](https://arxiv.org/abs/2510.02797), [HF space](https://huggingface.co/spaces/ASLP-lab/SongFormer)) | **New (Oct 2025), active**; code + weights + SongFormDB (14k songs) + SongFormBench (300 songs) released, CC-BY-4.0 | Structure boundaries + functional labels; **no beats/tempo** | Current SOTA on strict boundary detection; heavier (SSL audio encoders). Best quality option if structure alone matters; allin1 still wins on beats+structure in one package. |
| **MSAF** ([GitHub urinieto/msaf](https://github.com/urinieto/msaf)) | Legacy (2015-era), essentially unmaintained | Boundaries + flat labels via classic unsupervised algorithms (checkerboard, spectral clustering, CNMF…) | Useful as a baseline only; expect dependency archaeology. |
| **ruptures on SSM** | DIY approach (changepoint detection on a self-similarity matrix) | Boundaries only, no labels | Fine as a fallback; strictly worse than allin1/SongFormer for a product feature. |

**Recommendation:** **allin1** family is the right first choice (structure + beats + downbeats + BPM in one pass; `all-in-one-mlx` for Mac dev, upstream/`all-in-one-fix` on T4). Benchmark SongFormer later if label quality disappoints.

### B3. Does chord recognition on separated stems beat the full mix?

Evidence is **mixed and modest** — this is not a solved win:

- **Mitoma & Furuya, APSIPA ASC 2025** ([PDF](http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf)): exactly our setup — **htdemucs 4-stem** separation as preprocessing for **BTC**. Baseline 75.52% triads WCSR (485 songs: Isophonics 225, Robbie Williams 65, uspop2002 195; large 170-chord vocab; WCSR via mir_eval). They **double the amplitude of one stem and remix**; boosting the **"other" stem** (guitars/pianos) gave the best accuracy on all metrics (root, maj-min, triads, tetrads), correcting >2,000 misrecognized frames — but the overall gain is small (fractions of a point), and they note amplification of isolated notes sometimes *introduced* errors. Notably, the winning recipe **keeps the full mix and re-weights stems** rather than deleting drums/vocals.
- **Ko (UW-Madison project)** ([page](https://ko28.github.io/chord-transcription/)): models trained on source-separated audio performed **worse** on average, attributed to separation artifacts.
- **LLM chain-of-thought chord paper** ([arXiv:2509.18700](https://arxiv.org/html/2509.18700v1)): uses htdemucs to build drum-removed, drum+vocal-removed, and isolated-bass inputs as multiple views for reasoning — evidence that "bass stem for bass note, harmonic stems for chord quality" is a live research direction, but not packaged evidence of accuracy gains.
- **consonance-ACE** ([arXiv:2509.01588](https://arxiv.org/html/2509.01588v1)) improves inversions by *predicting* bass explicitly rather than by separating it — suggesting model-side fixes beat input-side surgery.

**Practical read:** since we already have stems, the cheap experiments are (1) chord model on `bass + other` remix (drums/vocals removed), and (2) APSIPA-style `mix + 2×other` re-weighting. Expect small deltas either way; measure before promising anything.

### B4. Key/BPM: keep librosa?

- **Key:** librosa has **no built-in key detection** ([librosa #366](https://github.com/librosa/librosa/issues/366)) — anything we do today is custom Krumhansl–Schmuckler on chroma. **essentia's `KeyExtractor`** is the best maintained drop-in (multiple profiles; its EDM-trained profiles reportedly outperform classic Krumhansl–Kessler on broad repertoire — [ref](https://arxiv.org/pdf/2605.06685)), and essentia now ships macOS arm64 wheels (May 2026). **But essentia is AGPL-3.0** — adopting it inside the app has license implications. Alternative: [libKeyFinder](https://github.com/ibsh/libKeyFinder) (C++, **GPL v3** — same problem). Verdict: keep the librosa-based custom key code unless key quality becomes a user-visible complaint; if it does, evaluate essentia in an isolated (server-side, AGPL-compliant) service or re-implement a better key profile (Temperley/EDMA profiles are published; the algorithm is ~50 lines on top of librosa chroma).
- **BPM/beats:** librosa's beat tracker is serviceable but dated; madmom's DBN tracker was the classic best and is now unmaintained. If we ship the structure feature, **allin1 gives beats/downbeats/tempo for free** at SOTA-ish quality — no separate swap needed. (For completeness: [beat_this](https://github.com/CPJKU/beat_this) (CPJKU) is the modern standalone beat tracker, but it inherits a madmom dependency issue for post-processing.)

---

## Unknowns that need local measurement

1. **Does our htdemucs vocal stem beat the mix for Whisper on *our* target songs?** Literature says "depends on mix density" (neutral on Jam-ALT, +4 pts WER on MUSDB). Our genre mix decides.
2. **Word-timestamp quality on singing** — nobody benchmarks word-level (only line-level in Jam-ALT). Whisper DTW timestamps vs. WhisperX wav2vec2 alignment on sustained/melismatic vowels is an open question; wav2vec2 aligners are speech-trained.
3. **Silero VAD vs. simple RMS gating on the vocal stem** — Silero is speech-trained and may drop sung sections; RMS-VAD needs a threshold tuned on our data.
4. **large-v2 vs large-v3 vs turbo vs Parakeet on lyrics** — the literature is split (v3 hallucinated more in several reports; Parakeet claims lyrics robustness but has no public lyrics WER).
5. **Rap and heavy-FX vocals** — no public benchmark found; needs a small curated test set.
6. **Chord accuracy delta from stem re-weighting** — APSIPA reports marginal gains on Beatles-era corpora; unknown on modern productions.
7. **BTC pretrained weights license** and **madmom model license (non-commercial?)** — must be verified before anything ships.
8. **allin1 install health on current Python/PyTorch on the T4 image** (NATTEN build + madmom-from-git are the known risks), and whether we can feed it our existing Demucs output to avoid double separation.
9. **mlx-whisper vs faster-whisper WER parity** — speed numbers are public; accuracy parity on music is assumed, not measured.
10. **Latency/VRAM budget on T4** for the combined pipeline (Demucs + ASR + allin1 + chords) per song.

---

## Suggested minimal local experiments

### Experiment A — Lyrics (1–2 days)

- **Data:** 10–15 songs with known lyrics: 5 from MUSDB18 test set (we have it; MUSDB-ALT provides transcripts), 5 personal picks spanning pop / rock / rap / heavy FX, plus 2 instrumentals (hallucination probes).
- **Run (grid):**
  - Inputs: {full mix, htdemucs vocal stem}
  - Models: {whisper large-v2, large-v3, large-v3-turbo} via mlx-whisper on the Mac; same via faster-whisper (+ Parakeet-TDT-0.6B) if the T4 is handy
  - Gating: {none, Silero VAD, RMS gate on vocal stem (e.g., drop segments where stem RMS < threshold)}
- **Measure:**
  - WER/CER per condition (normalize text; `jiwer` or Jam-ALT's evaluation code)
  - Insertion count on the 2 instrumentals (hallucination metric — should be ~0 after gating)
  - Spot-check word timestamps against audio for 3 songs (are highlighted words within ~±300 ms?)
  - Wall-clock per song on the M-series Mac
- **Decision gate:** ship-worthy if best condition reaches ≤ ~25% WER on the pop subset with near-zero instrumental hallucinations; pick stem-vs-mix and gating strategy from the grid.

### Experiment B — Chords + structure (1–2 days)

- **Data:** 10 songs from Isophonics Beatles (public chord + structure annotations) + 3 modern songs (qualitative only).
- **Run:**
  - Structure/beats: `all-in-one-mlx` on the Mac (and upstream `allin1` in a scratch T4 env to test install health). Record install friction honestly.
  - Chords: BTC pretrained (large-vocab checkpoint) and chord-extractor/Chordino on three inputs: {full mix, bass+other remix, mix with 2× "other" stem (APSIPA recipe)}.
- **Measure:**
  - Chords: WCSR (maj/min and triads) via `mir_eval.chord` against Isophonics labels, per input condition
  - Structure: boundary hit rate @0.5s/@3s and label accuracy via `mir_eval.segment`; sanity-check beats/BPM against tapped values
  - Wall-clock + peak memory per song, per tool
- **Decision gate:** structure feature is a go if allin1 boundaries look right on ≥7/10 songs and installs cleanly on at least one target platform. Chord feature is a go at "beta" quality if BTC hits ≥70% maj/min WCSR; adopt a stem-conditioned input only if it beats full mix by ≥1 pt.

---

## Source index

**Lyrics:** [arXiv:2506.15514](https://arxiv.org/abs/2506.15514) · [Jam-ALT](https://audioshake.github.io/jam-alt/) · [arXiv:2311.13987](https://arxiv.org/abs/2311.13987) · [AudioShake benchmark blog](https://www.audioshake.ai/post/new-benchmark-for-higher-quality-lyrics-transcription-from-audioshake-research) · [Music.AI benchmark](https://music.ai/blog/research/lyric-transcription-benchmark/) · [whisper.cpp Queen test](https://github.com/ggml-org/whisper.cpp/discussions/3074) · [openai/whisper #679](https://github.com/openai/whisper/discussions/679) · [Calm-Whisper](https://www.isca-archive.org/interspeech_2025/wang25b_interspeech.pdf) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [faster-whisper #843 (Silero on noise)](https://github.com/SYSTRAN/faster-whisper/issues/843) · [WhisperX](https://github.com/m-bain/whisperX) · [whispermlx](https://github.com/KalebJS/whispermlx) · [whisper-bench](https://github.com/naveedn/whisper-bench) · [mac-whisper-speedtest](https://github.com/anvanvan/mac-whisper-speedtest) · [large-v3-turbo card](https://huggingface.co/openai/whisper-large-v3-turbo) · [distil-large-v3.5](https://huggingface.co/distil-whisper/distil-large-v3.5) (+[timestamps discussion](https://huggingface.co/distil-whisper/distil-large-v3.5/discussions/6)) · [parakeet-tdt-0.6b-v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) · [Canary/Parakeet paper](https://arxiv.org/pdf/2509.14128) · [Modal batch ASR](https://modal.com/blog/fast-cheap-batch-transcription) · [AssemblyAI on Whisper hallucination](https://www.assemblyai.com/blog/comparing-universal-2-and-openai-whisper)

**Chords/structure:** [APSIPA 2025 stem-chord paper](http://www.apsipa.org/proceedings/2025/papers/APSIPA2025_P307.pdf) · [BTC repo](https://github.com/jayg996/BTC-ISMIR19) · [BTC paper](https://arxiv.org/abs/1907.02698) · [consonance-ACE](https://github.com/andreamust/consonance-ACE) ([paper](https://arxiv.org/abs/2509.01588)) · [LLM CoT chords](https://arxiv.org/html/2509.18700v1) · [Ko chord-transcription](https://ko28.github.io/chord-transcription/) · [madmom PyPI](https://pypi.org/project/madmom/) · [beat_this #9](https://github.com/CPJKU/beat_this/issues/9) · [autochord](https://github.com/cjbayron/autochord) · [crema](https://github.com/bmcfee/crema) · [essentia PyPI](https://pypi.org/project/essentia/) · [chord-extractor](https://github.com/ohollo/chord-extractor) · [allin1](https://github.com/mir-aidj/all-in-one) ([paper](https://arxiv.org/abs/2307.16425), [PyPI](https://pypi.org/project/allin1/)) · [all-in-one-mlx](https://libraries.io/pypi/all-in-one-mlx) · [all-in-one-fix](https://pypi.org/project/all-in-one-fix/) · [SongFormer](https://github.com/ASLP-lab/SongFormer) ([paper](https://arxiv.org/abs/2510.02797)) · [MSAF](https://github.com/urinieto/msaf) · [librosa #366 (no key detection)](https://github.com/librosa/librosa/issues/366) · [libKeyFinder](https://github.com/ibsh/libKeyFinder)
