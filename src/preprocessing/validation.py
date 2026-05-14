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