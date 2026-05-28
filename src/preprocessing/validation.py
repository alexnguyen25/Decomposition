"""
Audio file validation for the preprocessing pipeline.

This module loads candidate audio files with librosa, enforces duration rules,
and returns basic signal metadata. Validation failures raise domain-specific
exceptions from ``src.utils.exceptions`` so callers can handle user input,
corrupt files, and policy violations distinctly.
"""

import librosa
from pathlib import Path


from src.utils.exceptions import CorruptedFile, InvalidLength


def validAudio(file_path: Path) -> None:
    """
    Validate an audio file and return core audio properties.

    Attempts a full load via librosa, enforces duration bounds, and infers
    channel count from the loaded waveform shape.

    Args:
        file_path: Absolute or relative path to an audio file (any format
            supported by librosa).

    Raises:
        CorruptedFile: If librosa cannot read the file.
        InvalidLength: If duration is under 5 seconds or over 10 minutes.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception:
        raise CorruptedFile("The file can't be loaded")
    

    duration = librosa.get_duration(y=y, sr=sr)
    checkLength(duration)


def checkLength(duration) -> None:
    """
    Enforce minimum and maximum duration for accepted clips.

    Args:
        duration: Track length in seconds (typically from librosa).

    Raises:
        InvalidLength: If ``duration`` is under 5 seconds or over 600 seconds (10 minutes).
    """
    
    if duration < 5 or duration > 600:
        raise InvalidLength("The file length needs to be at least 5 seconds and under 10 minutes.")
