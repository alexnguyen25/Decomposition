"""
Source separation via Demucs (HTDemucs and related models).

Runs ``python3 -m demucs.separate`` as a subprocess on a preprocessed audio
file, then collects the four standard stem WAV paths Demucs writes under
``<output_dir>/<model>/<track_stem>/``. Intended to run after preprocessing
(see ``src.preprocessing.processing``) so inputs are normalized clips.

Typical layout::

    output_dir/
      htdemucs/
        my_track/
          vocals.wav
          drums.wav
          bass.wav
          other.wav
"""

from pathlib import Path
import subprocess
from typing import Dict

from src.utils.exceptions import DemucsFail, DemucsNotFound


def separate(
    file_path: Path,
    output_dir: Path,
    model: str = "htdemucs",
) -> Dict[str, Path]:
    """
    Separate an audio file into stems and return paths to each output WAV.

    Invokes Demucs, then maps stem names to files under
    ``output_dir / model / file_path.stem /``.

    Args:
        file_path: Path to the input audio file (e.g. a processed WAV).
        output_dir: Root directory passed to Demucs ``-o``; stems are nested
            under ``output_dir / model / <stem> /``.
        model: Demucs model name for ``-n`` (default ``htdemucs``).

    Returns:
        dict: Keys ``vocals``, ``drums``, ``bass``, ``other``; values are
        ``Path`` objects for each stem WAV.

    Raises:
        DemucsFail: If the Demucs subprocess exits with a non-zero status.
        DemucsNotFound: If ``python3`` or the ``demucs`` module cannot be run.
    """
    _run_demucs(file_path, output_dir, model)
    track_name = file_path.stem
    paths = _collect_stems(output_dir, track_name, model)

    return paths


def _run_demucs(
    file_path: Path,
    output_dir: Path,
    model: str,
) -> None:
    """
    Run Demucs separation as a subprocess.

    Args:
        file_path: Input audio path (converted to ``str`` for the CLI).
        output_dir: Output root for ``-o``.
        model: Model name for ``-n``.

    Raises:
        DemucsFail: On ``CalledProcessError`` (Demucs reported failure).
        DemucsNotFound: On ``FileNotFoundError`` (interpreter or module missing).

    Note:
        Uses ``capture_output=True`` so Demucs stdout/stderr are not printed
        to the terminal. ``check=True`` turns non-zero exit codes into exceptions.
    """
    command = [
        "python3", "-m", "demucs.separate",
        "-n", model,
        "-o", str(output_dir),
        str(file_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        raise DemucsFail("The file can't be processed")
    except FileNotFoundError:
        raise DemucsNotFound("Demucs can't be run because it's not found")


def _collect_stems(
    output_dir: Path,
    track_name: str,
    model: str,
) -> Dict[str, Path]:
    """
    Build a stem-name → file path map for Demucs output files.

    Args:
        output_dir: Root output directory (same as passed to ``_run_demucs``).
        track_name: Subfolder name under ``output_dir / model /`` (usually
            the input file stem).
        model: Demucs model folder name (e.g. ``htdemucs``).

    Returns:
        dict: ``vocals``, ``drums``, ``bass``, and ``other`` mapped to
        ``.wav`` paths under ``output_dir / model / track_name /``.
    """
    vocals = output_dir / model / track_name / "vocals.wav"
    drums = output_dir / model / track_name / "drums.wav"
    bass = output_dir / model / track_name / "bass.wav"
    other = output_dir / model / track_name / "other.wav"

    stem_paths = {
        "vocals": vocals,
        "drums": drums,
        "bass": bass,
        "other": other,
    }

    return stem_paths
