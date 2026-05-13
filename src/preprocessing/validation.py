"""
Audio file validation for the preprocessing pipeline.

This module loads candidate WAV files with librosa, enforces extension and
duration rules, and returns basic signal metadata. Validation failures raise
domain-specific exceptions from ``src.utils.exceptions`` so callers can
handle user input, corrupt files, and policy violations distinctly.
"""

import os

import librosa

from src.utils.exceptions import CorruptedFile, IncorrectExtension, InvalidLength


def validAudio(file_path):
    """
    Validate a WAV file and return core audio properties.

    Runs format checks, attempts a full load via librosa, enforces duration
    bounds, and infers channel count from the loaded waveform shape.

    Args:
        file_path: Absolute or relative path to a ``.wav`` file.

    Returns:
        dict: Keys ``sample_rate`` (Hz), ``channels`` (int), and ``duration``
        (seconds, float).

    Raises:
        IncorrectExtension: If the path does not end with ``.wav`` (case-insensitive).
        CorruptedFile: If librosa cannot read the file.
        InvalidLength: If duration is under 5 seconds or over 10 minutes.
    """
    checkFileFormat(file_path)

    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception:
        raise CorruptedFile("The file can't be loaded")
    

    duration = librosa.get_duration(y=y, sr=sr)
    checkLength(duration)

    channel_count = readChannelCount(y)

    return {"sample_rate": sr, "channels": channel_count, "duration": duration}


def checkFileFormat(file_path):
    """
    Ensure the file uses the allowed WAV extension.

    Args:
        file_path: Path to the candidate audio file.

    Raises:
        IncorrectExtension: If the suffix is not ``.wav`` (case-insensitive).

    Note:
        Corruption is not detected here; use ``validAudio`` (librosa load) for that.
    """
    
    filename, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext != '.wav':
        raise IncorrectExtension("The file is not a .wav")

def checkLength(duration):
    """
    Enforce minimum and maximum duration for accepted clips.

    Args:
        duration: Track length in seconds (typically from librosa).

    Raises:
        InvalidLength: If ``duration`` is under 5 seconds or over 600 seconds (10 minutes).
    """
    
    if duration < 5 or duration > 600:
        raise InvalidLength("The file length needs to be at least 5 seconds and under 10 minutes.")


def readChannelCount(y):
    """
    Infer number of audio channels from a librosa-loaded array ``y``.

    Args:
        y: Waveform as returned by ``librosa.load``: 1-D mono ``(n_samples,)``,
            or 2-D ``(n_channels, n_samples)``. For ``ndim > 2``, the code assumes
            a batch layout ``(..., channels, samples)`` and reads ``y.shape[1]``.

    Returns:
        int: Channel count (at least 1).
    """

    if y.ndim == 1:
        channels = 1
    elif y.ndim == 2:
        channels = y.shape[0]
    else:
        channels = y.shape[1] # For 3D batches structured as (tracks, channels, samples)
    
    return channels