"""Check whether a Demucs stem is basically empty using framed RMS."""

from pathlib import Path
from typing import Mapping

import librosa
import numpy as np

# TBD — calibrate on real Demucs stems
FRACTION_ABOVE_FLOOR = 0.05
STEM_FLOORS = {
    "vocals": 0.01,
    "drums": 0.01,
    "bass": 0.01,
    "other": 0.01,
}


def stem_energy(y):
    """Return framed RMS values for a waveform."""
    if y.ndim > 1:
        y = y.mean(axis=0)
    return librosa.feature.rms(y=y)[0]


def is_stem_silent(y, stem_name):
    """True if too few frames are above this stem's energy floor."""
    rms = stem_energy(y)
    floor = STEM_FLOORS[stem_name]
    loud_fraction = (rms > floor).mean()
    return loud_fraction < FRACTION_ABOVE_FLOOR


def check_stem_presence(stems: Mapping[str, Path]) -> dict[str, bool]:
    """Map each given stem to True if it's present (not near-silent)."""
    presence = {}
    for stem_name, path in stems.items():
        y, _ = librosa.load(path, sr=None, mono=False)
        presence[stem_name] = not is_stem_silent(y, stem_name)
    return presence
