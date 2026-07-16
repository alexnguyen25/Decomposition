"""
Feature extraction utilities.

This module extracts a small set of audio features from separated stems.

Conventions used by this project:
- The **drums** stem is used for tempo/BPM estimation.
- The **other** stem (everything except the main separated target, e.g. vocals)
  is used for key estimation and mel-spectrogram computation.

Primary entrypoint:
- `extract_features(stems)`: compute all supported features from a dict of stem
  file paths (at minimum containing `"drums"` and `"other"`).
"""

from pathlib import Path
from typing import Dict, Mapping, Union

import librosa
import numpy as np

from src.config import HOP_LENGTH, N_FFT, N_MELS, SR


def extract_mel_spectrogram(audio_path_other: Path) -> np.ndarray:
    """
    Compute a mel-scaled power spectrogram for an audio file.

    Args:
        audio_path_other: Path to an audio file readable by `librosa.load`.
            By project convention, this is typically the `"other"` stem.

    Returns:
        A 2D numpy array of shape `(n_mels, t)` containing the mel spectrogram.
        The exact shape depends on `librosa` defaults (e.g. hop length) and the
        audio duration.
    """
    y, _ = librosa.load(audio_path_other, sr=SR)

    S = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    return librosa.power_to_db(S, ref=np.max)


def extract_bpm(audio_path_drums: Path) -> float:
    """
    Estimate tempo (BPM) from an audio file.

    Args:
        audio_path_drums: Path to an audio file readable by `librosa.load`.
            By project convention, this should be the `"drums"` stem.

    Returns:
        Estimated tempo in beats-per-minute (BPM).
    """
    y, sr = librosa.load(audio_path_drums, sr=None)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    # librosa returns a scalar tempo for mono input (and sometimes a numpy scalar).
    return float(tempo)


def extract_key(audio_path: Path) -> str:
    """
    Estimate musical key and scale from an audio file using Essentia's Key algorithm.

    Loads mono audio, computes HPCP frames, and runs key estimation via
    ``KeyExtractor`` (which wraps the Key algorithm). The KeyExtractor requires
    mono audio signal.

    Args:
        audio_path: Path to a WAV or other format readable by Essentia's MonoLoader.

    Returns:
        str: Key and scale joined, e.g. ``"C# minor"`` or ``"F major"``.
    """
    import essentia.standard as es  # type: ignore[import-not-found]

    audio = es.MonoLoader(filename=str(audio_path))()
    key, scale, _strength = es.KeyExtractor()(audio)
    return f"{key} {scale}"


def extract_features(
    stems: Mapping[str, Path],
) -> Dict[str, Union[str, float, np.ndarray]]:
    """
    Extract all supported features from stem file paths.

    The minimum required stem keys are:
    - `"drums"`: used for BPM estimation
    - `"other"`: used for key estimation and mel spectrogram

    Args:
        stems: Mapping of stem name to audio file path.

    Returns:
        Dict containing:
        - `"key"` (`str`): estimated key + scale (e.g. `"C# minor"`)
        - `"bpm"` (`float`): estimated tempo in BPM
        - `"mel_spectrogram"` (`np.ndarray`): mel spectrogram array

    Raises:
        KeyError: If required keys (`"drums"`, `"other"`) are missing.
    """
    return {
        "key": extract_key(stems["other"]),
        "bpm": extract_bpm(stems["drums"]),
        "mel_spectrogram": extract_mel_spectrogram(stems["other"]),
    }
