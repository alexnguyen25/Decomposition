# Improving Classifier F1 — Research & Experiments
**Date:** July 21, 2026
**Role note:** All experiments were run by Claude in a session scratchpad as research. No `src/` code was changed and nothing was committed. Everything below is for Alex to implement.

> **Status: COMPLETE** (finalized July 22, 2026; second pass added §8). Headline: test macro-F1 **0.650 → 0.8045** across 11 experiments (champion: frozen BEATs embeddings + 12-second MLP head), plus a measured diagnosis and fix for the real-song over-prediction problem (stem-domain head fine-tune: real-song recall 0.585 → 0.659).

---

## 1. Where we started

- Confirmed baseline: the current checkpoint (`models/classifier.pt`, 2-conv CNN, Adam lr=1e-3, 25 epochs, no val split) scores **test macro-F1 0.6498** — reproduced locally on a fresh mel cache, matching the Colab eval from Day 16.
- Worst classes cluster by acoustic family: flute 0.263, accordion 0.272, clarinet 0.293 (wind/reed confusability, not sample count).
- Known deployment problem (Day 16/17): over-prediction of wind/brass/string instruments when classifying real Demucs "other" stems.

## 2. What the literature says is achievable

The 0.650 baseline is far below published results on OpenMIC-2018, which means the gap is recipe/architecture, not dataset difficulty:

| Approach | macro-F1 | Source |
|---|---|---|
| VGGish features + random forest (the dataset's own baseline) | **0.785** | Humphrey, Durand & McFee, ISMIR 2018 |
| VGG-style CNN from scratch, tuned | 0.801 | Koutini et al. 2021 |
| CP_ResNet (receptive-field regularized) | 0.809 | Koutini et al. 2021 |
| Shake-Shake CP_ResNet | **0.822** | Koutini et al. 2021 (best from-scratch) |
| PaSST / transformer models | ~0.84–0.87 (mAP, different metric) | Koutini et al. 2022 |

Metric caveat: the transformer line reports mAP, not macro-F1 — only compare against the first group.

Techniques ranked by expected gain (per the literature):
1. **Pretrained audio embeddings + shallow head** (VGGish ships with the dataset; PANNs CNN14 is stronger)
2. **Per-class decision thresholds tuned on a validation split** (macro-F1 is threshold-sensitive; requires a val split, which also fixes the Day-16 methodology objection)
3. **`pos_weight` / loss re-weighting** for class imbalance (recall on rare classes is where macro-F1 dies)
4. **Deeper CNN with BatchNorm** and receptive-field control
5. **SpecAugment** (time/freq masking)
6. **Mixup: skip it** — Koutini et al. explicitly found no gain on OpenMIC

## 3. Experimental setup (all local, Apple MPS)

- Full 20k-clip mel cache built locally (same `extract_mel_spectrogram`, byte-identical format).
- **Validation split: 15% of the train partition (seed 42), never touching test.** Early stopping and all threshold tuning use only this split.
- Metric: the project's own masked per-class F1 → macro-F1 (identical math to `evaluate.py::compute_metrics`).
- Every experiment reports test macro-F1 both at threshold 0.5 and with per-class thresholds tuned on the val split.

## 4. Results

| # | Experiment | Test macro-F1 @0.5 | @tuned thresholds | Train time |
|---|---|---|---|---|
| E0 | Current checkpoint (baseline) | **0.6498** | — | (25 ep, Colab) |
| E1 | Same tiny CNN + better recipe (input norm, AdamW+cosine, early stop) | 0.5638 | 0.6701 | ~40 min |
| E2 | 4-block CNN with BatchNorm (~394k params) + E1 recipe | 0.6989 | 0.7135 | ~45 min |
| E3 | E2 + SpecAugment | 0.6845 | 0.7046 | ~65 min |
| E4 | E3 + per-class `pos_weight` | **0.7442** | 0.7407 | 139 min |
| E5 | VGGish embeddings (mean+max pooled) + 2-layer MLP | 0.7462 | 0.7518 | 2.6 min |
| E6 | VGGish embeddings + per-class attention pooling (Gururani-style) | 0.7588 | 0.7631 | 1.7 min |
| E7 | PANNs CNN14 embeddings + MLP head | 0.7891 | **0.7915** | **0.2 min** |
| E8 | Ensemble (E4 + E6 + E7, averaged probabilities) | 0.7939 | **0.7943** | — |

Key observations:
- **Recipe alone is not enough (E1).** The tiny model with a modern recipe still lands *below* baseline at threshold 0.5 (0.564) and only slightly above it after threshold tuning (0.670). The 20k-parameter architecture is the bottleneck — no recipe rescues 64 channels of capacity.
- **Architecture is the big from-scratch lever (E2):** +0.04–0.06 over E1 from BatchNorm + depth alone.
- **SpecAugment *hurt* slightly (E3 vs E2: −0.01).** Honest negative result — the masking widths (16 mel bins / 40 frames, ×2 each) may be too aggressive for 10s weak-label clips, or need longer training to pay off. Koutini et al. saw gains with it, so parameterization is worth revisiting; don't cargo-cult it.
- **`pos_weight` was the biggest single CNN win (E4 vs E3: +0.06 @0.5).** Re-weighting positives by the per-class neg/pos ratio (capped at 8) directly attacks the rare-class recall problem where macro-F1 dies. Note E4's @0.5 beats its tuned number — the loss re-weighting already moves each class's operating point to roughly the F1-optimal spot, so threshold tuning has nothing left to fix (they're two mechanisms for the same correction).
- **E5:** +0.10 macro-F1 over baseline in 2.6 minutes of training, using the VGGish features already sitting in `data/openmic/openmic-2018/vggish/` (10×128 uint8 per clip; scale to [0,1], pool mean+max → 256-d, 2-layer MLP with BatchNorm+dropout, masked BCE).
- **E6:** attention pooling over the 10 VGGish timesteps beats fixed pooling (+1.3 points over E5). Each class learns *where in the clip* to listen — the multi-instance-learning view of OpenMIC's weak labels.
- **Pretrained features beat every from-scratch CNN** (E5/E6 > E4) at 1/50th the training cost — exactly what the literature predicted.
- **E7 is the headline: 0.789 → beats the original OpenMIC paper's VGGish+RF baseline (0.785), and the head trained in 12 seconds** (validation F1 peaked at epoch 1). AudioSet-pretrained CNN14 embeddings (2048-d) are simply a far better representation than anything trainable from 15k weakly-labeled clips. Deployment cost: running CNN14 at inference (~320 MB checkpoint) — a few seconds per song on CPU/MPS.
- **E8 (ensemble): 0.794, +0.144 over baseline** — averaging the three model families' probabilities adds a final half-point over E7 alone. Diminishing returns; only worth it if the pipeline can afford three forward passes.

## 5. Deployment reality check #1: the Demucs distribution shift, measured

Method: 80 OpenMIC test clips (each with ≥1 confirmed positive among the 16 pipeline-reported classes) were run through htdemucs. The baseline checkpoint classified both the full mix and the "other" stem; ground truth = OpenMIC confirmed labels, restricted to the 16 non-dead-weight classes.

Result — the shift is a **precision** problem, not a recall problem:

- Macro-F1 barely changes: mix 0.577 → stem 0.590 (per-class support here is tiny, so read as "no measurable degradation").
- **False positives on confirmed negatives: mix 18 → stem 28 (+56%)** on identical clips. Separated stems make the model fire on absent instruments more, even when it still finds what's really there.

Why day 17 looked much worse (10 FPs on one song): these are 10-second clips — one model call each. The real pipeline chops a full song into ~20 chunks and **max-pools** probabilities across them (`classifier.py::aggregate`). If a class has even a ~5% per-chunk FP rate, the probability that *some* chunk crosses 0.5 across 20 chunks approaches 1. Max-pooling converts occasional per-chunk noise into near-certain per-song false positives. Section 6 tests this directly on real songs (max vs mean aggregation).

## 6. Deployment reality check #2: real songs from the internet

Method: 14 CC-licensed tracks from the [MTG-Jamendo dataset](https://github.com/MTG/mtg-jamendo-dataset) (curated instrument tags mapped to OpenMIC classes, weak classes prioritized), full pipeline: download → Demucs → "other" stem → chunk into 431-frame windows → classify → aggregate chunk probabilities to a song-level decision.

**Caveats before the numbers:** Jamendo tags aren't exhaustive — a predicted-but-untagged instrument is only *probably* false ("extras" is an upper bound on FPs); a missed tagged instrument is a real miss. The sample also skews orchestral (several tracks share albums), so treat these as directional, not precise.

Tag-recall vs phantom instruments per track ("extras"), all at threshold 0.5:

| Model / aggregation | Tag recall | Extras per track |
|---|---|---|
| Baseline, **max**-pool (current `classifier.py`) | 0.793 | **6.71** |
| Baseline, mean-pool | 0.439 | 2.43 |
| E4 CNN, max-pool | **0.890** | 6.79 |
| E4 CNN, top-3-mean | 0.841 | 6.21 |
| E4 CNN, top-25%-mean | 0.780 | 5.36 |
| E4 CNN, mean-pool | 0.561 | 3.36 |
| E7 PANNs, max-pool | 0.622 | 3.29 |
| E7 PANNs, top-3-mean | 0.585 | 2.93 |
| E7 PANNs, mean-pool | 0.293 | 0.50 |

What this says:

1. **The day-16/17 over-prediction problem is reproduced and quantified:** the current pipeline (baseline + max-pool) invents ~7 instruments per song. Both models produce ~6.7–6.8 extras under max-pooling — so the FP explosion is **aggregation-driven, not model-driven**: with ~20 chunks per song, max-pooling turns any small per-chunk FP rate into a near-certain per-song FP (this compounds the +56% per-clip FP increase measured in Section 5).
2. **Aggregation sets a recall/precision dial:** max → recall-heavy; mean → precision-heavy; top-k in between. No aggregator alone gives both.
3. **Distribution shift hits the embedding model harder on recall:** E7, despite being far better on OpenMIC test, is very conservative on real Demucs stems (0.62 recall) — its head was trained and thresholded on full-mix embeddings. Model quality on the test set does **not** automatically transfer to the deployment domain.
4. **Classifying the stem beats classifying the mix** (e.g., E4 mean: 0.561 vs 0.451 on the mix) — separation genuinely isolates the target instruments; the pipeline design is right.
5. **The real fix is domain-matching, not aggregation:** re-tune per-class thresholds (and ideally re-train/fine-tune the head) on **Demucs-processed OpenMIC clips** so the training/calibration domain matches deployment. Section 5's machinery (run htdemucs over training clips, cache "other"-stem mels/embeddings) is exactly the recipe from Chiu et al. 2021.

## 7. Recommended implementation order

Ranked by measured value per unit of effort. Each step is independently shippable.

1. **Create a validation split** (15% of train, fixed seed; never touch test for any decision). Foundation for everything below — early stopping, threshold tuning, calibration. ~20 lines.
2. **PANNs CNN14 embeddings + small MLP head** — the single biggest win: **0.650 → 0.789** (+0.14). Precompute embeddings once (`data/openmic/panns_split01_{train,test}.npz` already exist from this research), train a 1-hidden-layer head with masked BCE in seconds. Inference on new songs needs CNN14 forward (~3 s/song on CPU/MPS). Checkpoint: Zenodo `Cnn14_mAP=0.431.pth`.
3. **Upgrade the from-scratch CNN to the E4 recipe** — 0.650 → 0.744, and it's the better *portfolio story* ("designed and trained my own CNN"): 4 conv blocks (32→64→128→256) each Conv3×3+BatchNorm+ReLU+MaxPool, GAP, dropout 0.3, linear; input normalized `(mel+40)/40`; AdamW lr 3e-4, wd 1e-4, cosine schedule; early stop on val macro-F1; **per-class `pos_weight` = clip(neg/pos, 1, 8)** in `BCEWithLogitsLoss`. Skip SpecAugment at first (measured −0.01 here); if revisiting, use gentler masks and longer training.
4. **Per-class thresholds tuned on the val split** — nearly free, +0.005–0.11 depending on model (huge for E1-style undertrained models, small once `pos_weight` already balances the operating point). This also resolves the day-16 methodology objection: the split exists now.
5. **Fix song-level aggregation in `classifier.py`** — replace max-pool with **top-3-mean** (or top-25%) over chunk probabilities: same code shape, strictly better precision/recall trade-off than max on real songs.
6. **Domain-match the calibration (the real over-prediction fix):** run Demucs over a few hundred OpenMIC train clips (code pattern in `data/f1_research/demucs_shift.py`), build a stem-domain validation set, re-tune per-class thresholds (and optionally fine-tune the head) on it. This attacks the measured +56% FP shift at its source rather than masking it.
7. **Optional polish:** attention pooling (E6, +1.3 pts over fixed pooling on VGGish — same idea applies to any per-timestep features) and the 3-model ensemble (0.794, +0.005 over E7). Diminishing returns.

**Portfolio framing tip:** the impressive artifact isn't just the final number — it's the measured ladder (0.650 → 0.744 from-scratch → 0.789 transfer learning → 0.794 ensemble), the honest negative results (SpecAugment hurt; mixup skipped on published evidence), and the deployment-shift study with a domain-matched fix. That's the difference between "trained a model" and "ran a research program."

## 7b. Reproducibility

All experiment code, checkpoints, and result JSONs from this research live in `data/f1_research/` (gitignored):
- `experiment.py` (E1–E4 + shared training/eval/threshold machinery), `vggish_head.py` (E5), `vggish_attention.py` (E6), `extract_panns.py`/`panns_head.py` (E7), `ensemble.py` (E8)
- `demucs_shift.py` (Section 5), `jamendo_prep.py`/`jamendo_eval.py`/`agg_study.py` (Section 6)
- `results.jsonl` (every experiment's full metrics incl. per-class F1 and tuned thresholds), `shift_results.json`, `jamendo_results.json`, `agg_study.json`
- Trained checkpoints `ckpt_*.pt`; PANNs embeddings in `data/openmic/panns_*.npz`; the full 20k-clip mel cache in `data/openmic/mel_cache/`

Same masked macro-F1 metric as `src/classification/evaluate.py` throughout; val split = 15% of train, seed 42.

## 8. Follow-up experiments (July 22 — second research pass)

New leaderboard after four more experiments:

| # | Experiment | Test macro-F1 @0.5 | @tuned | Train time |
|---|---|---|---|---|
| **E10** | **Frozen BEATs (iter3+ AS2M) embeddings + MLP head** | 0.7921 | **0.8045** | **0.2 min** |
| E11 | E10 + E7 probability ensemble | 0.7977 | 0.7996 | — |
| E8 | 3-model ensemble (E4+E6+E7) | 0.7939 | 0.7943 | — |
| E7 | Frozen PANNs CNN14 + head | 0.7891 | 0.7915 | 0.2 min |
| E9 | CNN14 fine-tuned end-to-end (blocks 4–6 + fc1) | 0.7874 | 0.7905 | 62 min |

- **E10 (new champion, 0.8045):** BEATs embeddings (Microsoft, ViT-B, 90M params) are the published OpenMIC frozen-feature SOTA (Quelennec et al., ICASSP 2024: 0.870 mAP), and they deliver here too — +0.013 over PANNs with the identical 12-second head recipe. Protocol: 16 kHz, two 5 s windows per clip, mean token pooling, mean over windows → 768-d. Code + checkpoint: `data/f1_research/beats/`; embeddings cached as `beats_split01_{train,test}.npz`.
- **E9 (honest negative result):** fine-tuning the CNN14 backbone (62 min) *ties* the frozen-embedding head (12 s) — 0.7905 vs 0.7915. On 12.7k weakly-labeled clips there is nothing to gain from end-to-end fine-tuning; keep backbones frozen.
- **E11 (negative):** averaging BEATs+PANNs probabilities (0.7996) *underperforms* BEATs alone with tuned thresholds — the weaker model dilutes the stronger one. Ensembles only helped when no single model dominated.
- **Stem-domain recalibration (deployment fix, measured):** fine-tuning the head on 300 Demucs-processed train clips (3× oversampled, 4 epochs) **raises real-song tag recall 0.585 → 0.659** (14 Jamendo tracks, top-3-mean) at +0.7 extras/track — even though it looks worse on the small 80-clip stem proxy eval (0.826 → 0.773; and threshold-only recalibration on 300 clips just overfits: skip it). Lesson: evaluate deployment fixes on the deployment distribution. Artifacts: `ckpt_E7_stem_recalib.pt`, `stem_recalib_results.json`.

**Revised recommendation:** the production classifier should be **frozen BEATs embeddings + MLP head** (0.8045), with a **stem-domain fine-tuned variant of that head for the real-song pipeline** (repeat the recalib-B recipe on BEATs embeddings of the 300 cached stems in `data/f1_research/stem_audio_cache/`). PANNs (E7) remains a fine fallback and the CNN14 checkpoint is still needed nowhere if BEATs ships. Update the Section 7 ladder accordingly: step 2 becomes "BEATs embeddings + head."

## 9. Learning resources

### Techniques you'd implement
- [SpecAugment (Park et al., 2019)](https://arxiv.org/abs/1904.08779) — original paper; implement freq+time masking, skip time-warp.
- [torchaudio augmentation tutorial](https://docs.pytorch.org/audio/stable/tutorials/audio_feature_augmentation_tutorial.html) — runnable `FrequencyMasking`/`TimeMasking` reference.
- [How Does Batch Normalization Help Optimization? (MIT)](https://gradientscience.org/batchnorm/) — best BN intuition; the "internal covariate shift" story is largely a myth. Pair with [d2l.ai §8.5](https://d2l.ai/chapter_convolutional-modern/batch-norm.html) for the train/eval-mode gotcha.
- [AdamW (Loshchilov & Hutter)](https://arxiv.org/abs/1711.05101) — why AdamW ≠ Adam+L2; cosine schedules.
- [d2l.ai §5.5 Generalization](https://d2l.ai/chapter_multilayer-perceptrons/generalization-deep.html) — validation splits and early stopping done right.
- [Thresholding Classifiers to Maximize F1 (Lipton et al.)](https://arxiv.org/abs/1402.1892) — theory behind per-class threshold tuning.
- [`BCEWithLogitsLoss` docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — `pos_weight` semantics (per-class neg/pos ratio).
- [Partial-label BCE (Durand et al., CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/papers/Durand_Learning_a_Deep_ConvNet_for_Multi-Label_Classification_With_Partial_Labels_CVPR_2019_paper.pdf) — the masked-loss idea behind OpenMIC's Y_mask.

### Pretrained embeddings / transfer learning
- [PANNs (Kong et al.)](https://arxiv.org/abs/1912.10211) + [audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) — CNN14 is the strongest Mac-friendly PyTorch backbone; frozen embeddings via `panns-inference`.
- [torchvggish](https://github.com/harritaylor/torchvggish) — PyTorch VGGish; produces the same 128-d embeddings the dataset ships with (needed to run VGGish-based models on *new* audio).
- [PaSST](https://arxiv.org/abs/2110.05069) + [repo](https://github.com/kkoutini/PaSST) — strongest results, heaviest to fine-tune; frozen-extractor first.
- [OpenL3](https://github.com/marl/openl3) — music-trained but TensorFlow-based; least convenient in this stack.

### OpenMIC-specific
- [OpenMIC-2018 paper](https://brianmcfee.net/papers/ismir2018_openmic.pdf) — dataset design and mask semantics.
- [cosmir/openmic-2018](https://github.com/cosmir/openmic-2018) — official baseline notebook (per-class logistic regression on VGGish): a correctness reference.
- [Gururani et al. 2019, attention for instrument recognition](https://arxiv.org/abs/1907.04294) — the E6 architecture.
- [Koutini et al., receptive-field regularization](https://arxiv.org/abs/2105.12395) + [repo](https://github.com/kkoutini/cpjku_dcase20) — why RF size, not just depth, drives audio CNN generalization.
- [Transfer learning & bias correction with audio embeddings](https://arxiv.org/abs/2307.10834) — recent OpenMIC study, relevant to the deployment shift.

### Deployment robustness / distribution shift
- [audiomentations](https://github.com/iver56/audiomentations) / [torch-audiomentations](https://github.com/iver56/torch-audiomentations) — gain/EQ/noise/pitch augmentation to make training data look like Demucs stems.
- [Source-separation-based data augmentation (Chiu et al.)](https://arxiv.org/pdf/2106.08703) — the core fix for the over-prediction problem: train on separated stems so train matches deployment.
- [Temperature scaling (Guo et al., 2017)](https://arxiv.org/abs/1706.04599) — calibrating confidences on a stem-like validation set.

### General
- [Karpathy: A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — read before the next training run; the checklist mindset (verify data pipeline, overfit one batch, add complexity gradually) is the meta-lesson of this whole exercise.
- [Valerio Velardo — The Sound of AI](https://valeriovelardo.com/the-sound-of-ai/) and [musicinformationretrieval.com](https://musicinformationretrieval.com/) — audio-DL fundamentals.
- [Deep Learning for MIR tutorial (Choi et al.)](https://arxiv.org/abs/1709.04396) — broader MIR context.


---

## 10. Corrections from implementation (September 2, 2026)

Everything above was measured in a research scratchpad. Implementing it in
`src/` and re-measuring changed two conclusions. Both corrections are in the
repo's favour — they replace an assumption with a measurement — but §8 should
be read with them.

**Stem-domain recalibration did not replicate on BEATs.** §8 recommends
shipping `ckpt_E10_stem_recalib.pt`, extrapolating from the PANNs result
(real-song recall 0.585 → 0.659). That extrapolation was never verified:
`stem_recalib.py` (which wrote `stem_recalib_results.json`) recalibrates the
**E7/PANNs** head, while `beats_stem_recalib.py` printed its Jamendo numbers to
stdout and saved no results file. Re-running both BEATs heads over the same 14
Jamendo tracks:

| head | micro-recall | extras/track | mean F1 |
|---|---|---|---|
| `e10` | 0.573 | 2.36 | 0.562 |
| `stem_recalib` | 0.585 | 2.21 | 0.579 |

A one-tag difference (47/82 vs 48/82). On 5 further held-out Jamendo tracks
`e10` scored higher; pooled over all 19, both land at mean F1 **0.524**. The
PANNs lift (+0.073 recall) did not transfer to a stronger backbone (+0.012).
**`e10` ships**, because the two are indistinguishable and `e10` is the head
that is actually reproducible — `stem_audio_cache/` (the 380 Demucs clips that
trained the recalibrated variant) no longer exists, so that checkpoint cannot
be retrained or re-verified.

**Top-3-mean aggregation is a precision/recall trade, not the over-prediction
fix.** §6 attributes the real-song false-positive explosion to max-pooling.
Measured directly on the 14-track set (`AGGREGATION_TOP_K=1` is arithmetically
identical to max-pooling):

| backend | aggregation | P | R | F1 | preds/track |
|---|---|---|---|---|---|
| BEATs | max-pool | 0.558 | 0.670 | 0.555 | 6.7 |
| BEATs | top-3-mean | 0.595 | 0.627 | 0.562 | 5.7 |
| CNN | max-pool | 0.410 | 0.757 | 0.513 | 10.9 |
| CNN | top-3-mean | 0.427 | 0.707 | 0.509 | 9.9 |

Roughly F1-neutral, and slightly *negative* for the CNN. The CNN still predicts
~10 instruments per track under either scheme, so over-prediction is the model,
not the pooling. Top-3-mean ships for the precision gain, which matters when a
wrong instrument is rendered on screen — not as a fix.

**Reproduce:** `python scripts/eval_openmic.py` and
`python scripts/eval_real_songs.py`.

**The Gemini comparison was re-scored against the shipped head.** §6's headline
(0.875 vs 0.598) is the **E7/PANNs** arm, not what ships. Re-scoring Gemini's
cached per-clip predictions against a BEATs arm on the same 100 clips:

| arm | masked macro-F1 |
|---|---|
| BEATs + head @tuned (shipped) | 0.8517 |
| BEATs + head @0.5 | 0.8515 |
| `gemini-flash-latest` | 0.5981 |
| E7 PANNs @tuned (cached) | 0.8751 |

Gemini's number reproduces to four decimals, confirming the scoring matches.
Note that E7 outscores BEATs *on this 100-clip subset* while losing on the full
5,085-clip test set (0.7915 vs 0.8045) — 100 clips is simply too few to order
two close models, which is worth remembering before quoting either figure.
`gemini-pro-latest` ran on only 50 of the 100 clips and is excluded.

**Reproduce:** `python scripts/eval_vs_gemini.py` (no API calls).

