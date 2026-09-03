---
title: Decomposition
emoji: 🎛️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
---

# Decomposition — demo Space

Song → 4 Demucs stems + instrument detection (PANNs CNN14 + custom head,
OpenMIC-2018, test macro-F1 0.79) + BPM/key.

## Deploying this Space (steps for Alex)

1. Create a free account at huggingface.co, then **New Space** → Gradio → CPU
   Basic (free) — or ZeroGPU (free for 2 personal Spaces; add
   `import spaces` and `@spaces.GPU(duration=120)` on `analyze`).
2. Copy into the Space repo: `app.py`, `requirements.txt`, this `README.md`,
   plus model files:
   - `Cnn14_mAP=0.431.pth` (from `data/f1_research/`)
   - `e7_head.pt` (rename `data/f1_research/ckpt_E7_panns_head.pt`)
   - `class-map.json` (from `data/openmic/openmic-2018/`)
   Use `git lfs` for the .pth (320 MB): `git lfs track "*.pth" "*.pt"`.
3. `git push` — the Space builds and goes live at
   `https://huggingface.co/spaces/<user>/decomposition`.
4. Note: essentia isn't in requirements (heavy build on Spaces); key shows
   "n/a" unless you add `essentia` and it builds successfully.
5. Add 2–3 example songs (CC-licensed) with `gr.Examples` + cached outputs so
   reviewers get one-click results (see portfolio checklist).
