"""
Audio file validation for the preprocessing pipeline.

This module loads candidate audio files with librosa, enforces duration rules,
and returns basic signal metadata. Validation failures raise domain-specific
exceptions from ``src.utils.exceptions`` so callers can handle user input,
corrupt files, and policy violations distinctly.
"""

import os

import librosa
from pathlib import Path


from src.utils.exceptions import CorruptedFile, IncorrectExtension, InvalidLength

# Known non-audio extensions; anything else is allowed through to librosa.
_NON_AUDIO_EXTENSIONS = frozenset(
    {
        ".txt",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
        ".csv",
        ".md",
        ".rtf",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".webp",
        ".svg",
        ".ico",
        ".heic",
        ".html",
        ".htm",
        ".css",
        ".js",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".zip",
        ".tar",
        ".gz",
        ".bz2",
        ".rar",
        ".7z",
        ".exe",
        ".dll",
        ".so",
        ".dylib",
        ".app",
        ".deb",
        ".rpm",
        ".py",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".rs",
        ".go",
        ".rb",
        ".php",
        ".mp4",
        ".avi",
        ".mkv",
        ".mov",
        ".wmv",
        ".webm",
        ".flv",
        ".m4v",
    }
)


def validAudio(file_path: Path) -> None:
    """
    Validate an audio file and return core audio properties.

    Attempts a full load via librosa, enforces duration bounds, and infers
    channel count from the loaded waveform shape.

    Args:
        file_path: Absolute or relative path to an audio file (any format
            supported by librosa).

    Raises:
        IncorrectExtension: If the path has a known non-audio extension.
        CorruptedFile: If librosa cannot read the file.
        InvalidLength: If duration is under 10 seconds or over 10 minutes.
    """
    checkFileFormat(file_path)

    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception:
        raise CorruptedFile("The file can't be loaded")

    duration = librosa.get_duration(y=y, sr=sr)
    checkLength(duration)


def checkFileFormat(file_path: Path) -> None:
    """
    Reject paths whose extension is a known non-audio type.

    Audio formats are not whitelisted; unknown or missing extensions are allowed
    and validated later by ``librosa.load``.

    Args:
        file_path: Path to the candidate audio file.

    Raises:
        IncorrectExtension: If the suffix is a known non-audio extension.
    """
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext in _NON_AUDIO_EXTENSIONS:
        raise IncorrectExtension("The file is not an audio file")


def checkLength(duration) -> None:
    """
    Enforce minimum and maximum duration for accepted clips.

    Args:
        duration: Track length in seconds (typically from librosa).

    Raises:
        InvalidLength: If ``duration`` is under 10 seconds or over 600 seconds (10 minutes).
    """

    if duration < 10 or duration > 600:
        raise InvalidLength(
            "The file length needs to be at least 10 seconds and under 10 minutes."
        )
