# Dev Journal — Day 8
**Date:** June 9, 2026
**Project:** Decomposition

---

## What We've Done Today

- Took a two week break and got caught back up to speed on the project
- Researched CNN architecture end to end — everything needed before writing any classification code
- Built intuition for Conv2D, ReLU, MaxPool, flatten, dense layers, and sigmoid output through interactive visualizers
- Clarified the distinction between architecture and training
- Settled remaining open decisions: sigmoid vs softmax, where chunking lives, why CNN over other model types

---

## What We Built

Nothing coded today — pure research and architecture design session. The classification folder remains empty. All decisions made today feed directly into Day 9 and 10 implementation.

---

## Key Concepts Learned

**Conv2D and feature maps**
A filter is a small grid of weights (e.g. 3×3) that slides across the input, computing a dot product at every position. The output grid of those dot products is a feature map — it answers "how strongly did this filter's pattern appear at each location." A real conv layer runs many filters simultaneously (32, 64, etc.), each producing its own feature map. Those maps stack into a 3D tensor: depth = number of filters (channels), height and width = spatial dimensions of the input.

**What the shape numbers mean**
A shape like `(64, 32, 107)` means 64 channels stacked, each a 32×107 grid. The 3×3 filter size never changes — it's not one of these numbers. Height and width shrink through MaxPool; channels grow through more filters.

**ReLU**
The dot product inside a filter produces any real number — positive means the pattern is present, negative means the opposite of the pattern is present. ReLU zeros out negatives, keeping only the detections. Also prevents vanishing gradients during training.

**MaxPool**
Chops the feature map into 2×2 blocks and keeps only the maximum value in each block. Throws away exact position, keeps whether the pattern appeared in that region. Halves height and width each time it's applied. Builds translation invariance — a guitar chord at frame 3 vs frame 4 lands in the same block and gives the same answer.

**Flatten**
Does no math. Unrolls the entire 3D tensor into a single 1D vector by laying every value end to end. Length = depth × height × width. Converts the 3D structure into something a dense layer can read.

**Dense layer and the mapping to 20 instruments**
Each of the 20 outputs computes a separate weighted sum over every input value. Every output has its own set of weights — one per input value. If the flattened vector has 219,136 values and there are 20 outputs, the dense layer has 20 × 219,136 ≈ 4 million weights. The guitar output has its own 219,136 weights. Piano has a completely different set. The right weights are learned during training — they start random and get adjusted through backpropagation.

**Why raw scores aren't already 0 to 1**
The dense layer is a weighted sum — it produces any real number. Sigmoid squashes each of the 20 scores independently to a 0–1 probability. You threshold at 0.5 (or wherever) to call an instrument present or absent. During training, raw scores (logits) go directly into `BCEWithLogitsLoss` without applying sigmoid first — more numerically stable.

**Sigmoid vs softmax**
Softmax forces all outputs to sum to 1 — it's designed for "pick exactly one." Multi-label classification requires each output to be independent. Sigmoid applies independently to each score. Multi-label always uses sigmoid.

**Architecture vs training**
Architecture defines the structure of the network — what layers exist, how data flows, what shape everything is at each step. Weights are initialized randomly. Training is the separate process of adjusting those weights using labeled examples and a loss function until the network learns to predict correctly. Everything learned today is architecture. Training is Day 10.

**Why CNN over other approaches**
- MLP — destroys spatial structure by flattening immediately, needs far more data
- RNN/LSTM — models temporal order, which doesn't matter for instrument presence detection
- CRNN — valid but overkill for MVP
- Transformer — state of the art but data hungry and complex to implement
- CNN — designed for 2D spatial inputs, well understood, standard baseline for audio classification

**Chunking decision**
Feature extraction returns the full song spectrogram `(128, t)`. Chunking into 10-second windows lives in the classification stage, not in feature extraction. Feature extraction stays clean and general. Predictions across chunks get aggregated (max pooling) for a single song-level output.

---

## Struggles

- Conv filter intuition took time — specifically how a 3×3 filter detects patterns larger than itself (it fires repeatedly along the pattern, building a streak in the feature map that later layers detect)
- The shape `(64, 32, 107)` — initially misread 32 as "the number of values in the filter" rather than the height of the feature map
- Flatten was fuzzy — needed a visualizer to make it concrete that it's pure reshaping with no math
- How the dense layer actually maps to 20 outputs — clicked once the weighted sum framing landed: each output is its own independent weighted sum over all input values, with its own learned weights

---

## Action Items for Next Session

- [ ] OpenMIC-2018 data loader — load audio clips and labels
- [ ] Understand `y_mask` and masked loss before touching any training code
- [ ] Implement the CNN architecture in PyTorch
- [ ] Write the training loop with `BCEWithLogitsLoss` and masked loss

---

## Remaining Roadmap

| Day | Stage |
|-----|-------|
| 9 | OpenMIC data loader + weak label handling |
| 10 | CNN implementation + training loop |
| 11 | Training run + evaluation |
| 12 | JSON output + full pipeline integration in `main.py` |
| 13 | Buffer — bugs, cleanup, README, MVP done |