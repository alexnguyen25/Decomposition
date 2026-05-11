# Dev Journal — Day 1
**Date:** May 10, 2026  
**Project:** Decomposition

---

## What I'm Building

An audio analysis pipeline that takes a music file as input and outputs:
- 4 separated, playable stems (vocals, drums, bass, other)
- Instrument labels identifying what is present in the "other" stem
- Basic song characteristics (BPM, key)

---

## The Pipeline

```
INPUT: audio file (mp3/wav)
         ↓
PREPROCESSING
- Format conversion to wav
- Stereo to mono if needed
- Sample rate normalization
- Duration/corruption checks
         ↓
SOURCE SEPARATION (Demucs inference)
- Outputs 4 stems: vocals, drums, bass, other
         ↓
FEATURE EXTRACTION (Librosa)
- Mel spectrogram on "other" stem
- Basic features on all stems (BPM, key)
         ↓
INSTRUMENT CLASSIFICATION (custom ML model)
- Input: mel spectrogram of "other" stem
- Output: instrument labels + confidence scores
- Trained on: OpenMIC-2018
         ↓
OUTPUT (structured JSON)
- 4 playable stem files
- Instrument labels for "other" stem
- BPM, key, basic song info
```

---

## Key Concepts Learned

**Audio representations**
- Raw audio is a waveform — air pressure measurements over time (44,100 per second at 44.1kHz)
- A spectrogram converts audio from the time domain to the frequency domain — showing which frequencies are present at which points in time
- A mel spectrogram scales frequencies to match human perception — low frequencies get more resolution, high frequencies are compressed
- MFCCs (Mel-Frequency Cepstral Coefficients) capture the texture/timbre of audio at a moment in time — useful for distinguishing instruments that play the same note but sound different

**Source separation**
- Demucs (Meta) is a hybrid transformer model that separates audio into 4 stems: vocals, drums, bass, other
- It processes both the waveform and spectrogram in parallel — the "hybrid" approach
- Uses a U-Net architecture: compresses input down then expands back up, with shortcuts connecting both sides
- We use Demucs for inference only — not training it from scratch

**ML fundamentals**
- Training: feeding a model data and adjusting weights to reduce loss
- Inference: running a pretrained model on new inputs to get predictions
- Loss: a measure of how wrong the model's predictions are
- The training loop: predict → measure loss → adjust weights → repeat

**Instrument classification**
- A mel spectrogram is essentially an image (time x frequency x intensity) — so image classification techniques like CNNs apply to audio
- Our classifier is a multi-label classifier — one stem can contain multiple instruments
- Distribution shift: a model trained on isolated instruments may fail on mixed audio — reason we chose OpenMIC-2018 over IRMAS

---

## Key Decisions Made

**Use Demucs for source separation**
- Well-documented, reliable, actively maintained by Meta
- Separates into 4 stems: vocals, drums, bass, other
- We use inference only — not training from scratch

**Classify instruments in the "other" stem**
- Demucs already labels vocals/drums/bass
- "Other" is the catch-all for guitar, piano, strings, synths, etc.
- Our classifier identifies what's inside that mixed stem
- Limitation: we identify presence, not isolate into separate tracks

**Use OpenMIC-2018 as training data**
- 20,000 examples, 10-second excerpts, 20 instrument classes
- Contains instruments in a mix — matches real-world input
- IRMAS has isolated single-instrument recordings — closer to clean but mismatches our mixed input (distribution shift)
- OpenMIC is the more defensible choice for our use case

---

## Datasets Researched

| Dataset | Size | Labels | Notes |
|---|---|---|---|
| OpenMIC-2018 | 20k clips | 20 instrument classes, weak labels | Instruments in a mix — chosen for training |
| IRMAS | ~9.5k clips | 11 instrument classes | Isolated instruments — useful reference, not chosen |
| GTZAN | ~1k clips | 10 genre classes | Genre classification only — not relevant to our problem |

---

## Tools & Libraries

| Tool | Purpose |
|---|---|
| Demucs | Source separation (pretrained inference) |
| Librosa | Feature extraction (mel spectrograms, BPM, key) |
| OpenMIC-2018 | Training data for instrument classifier |
| Python | Primary language |

---

## What I Still Need to Figure Out

- Exact CNN architecture for the instrument classifier
- How to handle `y_mask` weak labels in OpenMIC-2018 during training
- Evaluation strategy — what metrics to use (confusion matrix, precision, recall)
- How long Demucs inference takes on a typical song — latency matters
- Folder structure and project setup

---

## Action Items for Next Session

- [ Done ] Create GitHub repo
- [ ] Set up Python virtual environment and folder structure
- [ ] Download OpenMIC-2018 from Zenodo
- [ ] Install Demucs and run it on a local audio file
- [ ] Install Librosa and extract a mel spectrogram from a stem

---

## Resources

- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [OpenMIC-2018 Zenodo](https://zenodo.org/record/1432913)
- [IRMAS Dataset](https://www.upf.edu/web/mtg/irmas)