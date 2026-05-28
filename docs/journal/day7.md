# Dev Journal — Day 7
**Date:** May 27, 2026
**Project:** Decomposition

---

## What We've Done Today

- Removed the wav-only restriction since we'd do that later anyways, but included restriction for only audio files
- built the feature_extraction.py
- Installed torchcodec so demucs could use like mp3 and other file formats
- We use sr=None on librosa.load() calls since stems out of Demucs are already 44,100 Hz
- Used Essentia for key detection instead of implementing Krumhansl-Schmuckler manually — faster for MVP, clean output
- not gonna write unittests since we did smoke tests and it works and it's pretty simple running on already processed filtered data

---

## What We Built

Built feature_extraction.py — four functions:

extract_mel_spectrogram(audio_path_other) — loads the "other" stem with Librosa, returns a (128, t) mel spectrogram array
extract_bpm(audio_path_drums) — loads the drums stem, runs librosa.beat.beat_track(), returns tempo as a float. Used drums because it gives the clearest onset signal for beat tracking.
extract_key(audio_path) — loads mono audio via Essentia's MonoLoader, runs KeyExtractor, returns a string like "C# minor"
extract_features(stems) — public API, takes the stems dict from separate(), returns a dict with key, bpm, and mel_spectrogram

---

## Key Concepts Learned

for instrument classification, the number of frequency bands (n_mels) is common 128. with more it does provide more info, but creates background noise, computational cost, and overfitting in ml models

librosa.beat.beat_track -  we use for getting the bpm
returns
tempofloat [scalar, non-negative] or np.ndarray
beatsnp.ndarray
Krumhansl-Schmuckler key-finding algorithm
works by correlating your chroma vector against predefined profiles for all 24 keys (12 major, 12 minor) and picking the best match.

---

## Struggles

Not really just a lot of research today

---

## Action Items for Next Session

CNN Architecture Research
