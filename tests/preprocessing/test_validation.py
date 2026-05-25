"""
Unit tests for ``src.preprocessing.validation``.

Exercises ``validAudio`` and the domain exceptions raised for bad inputs:
``IncorrectExtension``, ``CorruptedFile``, and ``InvalidLength``. Librosa is
mocked so no real WAV files are required.

Accepted policy (see ``validation.py``):
    - Extension: ``.wav`` only (case-insensitive, e.g. ``.WAV``)
    - Duration: at least 5 seconds and at most 600 seconds (10 minutes)

Run:
    python3 -m pytest tests/preprocessing/test_validation.py -v

Mocking notes:
    - ``return_value`` on ``librosa.load`` / ``get_duration`` simulates a
      successful read and a known duration.
    - ``side_effect`` on ``librosa.load`` simulates decode failures mapped to
      ``CorruptedFile``.
    - Extension checks run before load; incorrect-extension tests need no patch.
"""

import unittest
from unittest.mock import patch

import numpy as np

from src.preprocessing.validation import validAudio
from src.utils.exceptions import CorruptedFile, IncorrectExtension, InvalidLength


class TestValidAudio(unittest.TestCase):
    """Tests for ``validAudio`` and its validation rules."""

    @patch("src.preprocessing.validation.librosa.get_duration")
    @patch("src.preprocessing.validation.librosa.load")
    def test_happy_path_valid_wav_in_duration_range(self, mock_load, mock_get_duration):
        """
        Happy path: ``.wav`` path, load succeeds, duration within 5–600 s.

        Does not raise; confirms ``librosa.load`` and ``get_duration`` were used.
        """
        mock_load.return_value = (np.zeros(5292000, dtype=np.float32), 44100)
        mock_get_duration.return_value = 120.0

        validAudio("fake.wav")

        mock_load.assert_called_once_with("fake.wav", sr=None)
        mock_get_duration.assert_called_once()

    @patch("src.preprocessing.validation.librosa.load")
    def test_corrupted_file_raises_corrupted_file(self, mock_load):
        """
        Any exception from ``librosa.load`` is wrapped as ``CorruptedFile``.

        Uses ``side_effect`` to simulate a decode/runtime failure.
        """
        mock_load.side_effect = RuntimeError("decode failed")

        with self.assertRaises(CorruptedFile):
            validAudio("broken.wav")

    def test_incorrect_extension_raises_incorrect_extension(self):
        """Non-``.wav`` extensions (e.g. ``.mp3``) raise ``IncorrectExtension`` before load."""
        with self.assertRaises(IncorrectExtension):
            validAudio("fake.mp3")

    @patch("src.preprocessing.validation.librosa.get_duration")
    @patch("src.preprocessing.validation.librosa.load")
    def test_uppercase_wav_extension_is_accepted(self, mock_load, mock_get_duration):
        """
        Extension check is case-insensitive; ``.WAV`` is treated like ``.wav``.
        """
        mock_load.return_value = (np.zeros(5292000, dtype=np.float32), 44100)
        mock_get_duration.return_value = 120.0

        validAudio("fake.WAV")

        mock_load.assert_called_once_with("fake.WAV", sr=None)

    @patch("src.preprocessing.validation.librosa.get_duration")
    @patch("src.preprocessing.validation.librosa.load")
    def test_too_short_duration_raises_invalid_length(self, mock_load, mock_get_duration):
        """Duration under 5 seconds raises ``InvalidLength``."""
        mock_load.return_value = (np.zeros(176400, dtype=np.float32), 44100)
        mock_get_duration.return_value = 4.0

        with self.assertRaises(InvalidLength):
            validAudio("short.wav")

    @patch("src.preprocessing.validation.librosa.get_duration")
    @patch("src.preprocessing.validation.librosa.load")
    def test_too_long_duration_raises_invalid_length(self, mock_load, mock_get_duration):
        """Duration over 600 seconds raises ``InvalidLength``."""
        mock_load.return_value = (np.zeros(26504100, dtype=np.float32), 44100)
        mock_get_duration.return_value = 601.0

        with self.assertRaises(InvalidLength):
            validAudio("long.wav")


if __name__ == "__main__":
    unittest.main()
