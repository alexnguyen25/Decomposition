# Dev Journal - Day 3
**Date** May 12, 2026
**Project:** Decomposition

---

## What we've done today
- We wrote the 2 proprocessing files: processing.py and validation.py
- validation.py — validates an incoming WAV file before it touches the pipeline. Checks format, checks for corruption by attempting to load with Librosa, and enforces a 5 second minimum and 10 minute maximum duration. Raises custom exceptions for each failure type.
- processing.py — transforms a validated file to match what Demucs expects. Resamples to 44,100 Hz if needed, downmixes to stereo if channels exceed 2, saves the result to data/processed/, and returns the output path.
- We also created the exception classes CorruptedFile, IncorrectExtension, InvalidLength

## Key Concepts Learned
- We want to use WAV over MP3, because MP3 is lossy and has artifacts that cause distribution shifts in Demucs
- We use 44,100 Hz because Demucs was trained on this so the wrong sample rate means the model hears the audio at the wrong speed
- We use stereo because Demucs uses phase differences between channels for separation

## Action Items for Next Session
- Write unit tests for both modules of the preprocessing