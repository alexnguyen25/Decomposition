"""
Unit tests for ``src.preprocessing.processing``.

Covers sample-rate normalization, multi-channel downmixing, and the full
``process_audio`` pipeline. External I/O (librosa load/resample, soundfile
write) is mocked so tests run quickly without real audio files on disk.

Run:
    python3 -m pytest tests/preprocessing/test_preprocessing.py -v

Mocking notes:
    Patches target the symbol where it is used (``src.preprocessing.processing``),
    not where librosa/soundfile are defined. ``return_value`` simulates successful
    calls; assertions verify call arguments (e.g. ``target_sr=44100``).
"""

import os
import unittest
from unittest.mock import patch

import numpy as np

from src.preprocessing import processing
from src.preprocessing.processing import (
    normalizeChannels,
    normalizeSampleRate,
    process_audio,
)


def _expected_processed_dir():
    """Return ``data/processed`` path using the same root logic as ``process_audio``."""
    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(processing.__file__))
    )
    return os.path.join(project_root, "data", "processed")


class TestNormalizeSampleRate(unittest.TestCase):
    """Tests for ``normalizeSampleRate`` (resampling to 44.1 kHz)."""

    @patch("src.preprocessing.processing.librosa.resample")
    def test_resamples_to_44100_when_source_rate_differs(self, mock_resample):
        """
        When native sample rate is not 44100, ``librosa.resample`` is invoked
        with ``target_sr=44100`` and its return value is passed through.
        """
        y = np.zeros(1000, dtype=np.float32)
        sr = 22050
        mock_resample.return_value = np.zeros(2000, dtype=np.float32)

        result = normalizeSampleRate(y, sr)

        mock_resample.assert_called_once_with(y, orig_sr=sr, target_sr=44100)
        np.testing.assert_array_equal(result, mock_resample.return_value)

    @patch("src.preprocessing.processing.librosa.resample")
    def test_skips_resample_when_already_44100(self, mock_resample):
        """Input already at 44100 Hz is returned unchanged; resample is not called."""
        y = np.zeros(1000, dtype=np.float32)

        result = normalizeSampleRate(y, 44100)

        mock_resample.assert_not_called()
        np.testing.assert_array_equal(result, y)


class TestNormalizeChannels(unittest.TestCase):
    """Tests for ``normalizeChannels`` (layouts with more than two channels)."""

    def test_downmixes_more_than_two_channels_to_stereo(self):
        """
        A channel-major array with four channels ``(4, n_samples)`` is reduced
        to stereo by keeping the first two rows ``y[:2, :]``.
        """
        y = np.zeros((4, 1000), dtype=np.float32)

        result = normalizeChannels(y)

        self.assertEqual(result.shape, (2, 1000))
        np.testing.assert_array_equal(result, y[:2, :])

    def test_leaves_stereo_unchanged(self):
        """Two-channel input is not modified."""
        y = np.zeros((2, 1000), dtype=np.float32)

        result = normalizeChannels(y)

        np.testing.assert_array_equal(result, y)


class TestProcessAudio(unittest.TestCase):
    """Integration-style tests for ``process_audio`` with mocked I/O."""

    @patch("src.preprocessing.processing.sf.write")
    @patch("src.preprocessing.processing.librosa.resample")
    @patch("src.preprocessing.processing.librosa.load")
    def test_process_audio_resamples_downmixes_and_saves(
        self, mock_load, mock_resample, mock_write
    ):
        """
        End-to-end: load 4ch @ 22.05 kHz, resample to 44.1 kHz, downmix to stereo,
        write under ``data/processed`` as ``<stem>_processed.wav``.

        Asserts:
            - ``librosa.load`` called with ``sr=None``
            - ``librosa.resample`` uses ``target_sr=44100``
            - ``sf.write`` receives stereo ``(2, n)`` at 44100 Hz
            - Output path is ``data/processed/my_track_processed.wav``
        """
        mock_load.return_value = (np.zeros((4, 1000), dtype=np.float32), 22050)
        mock_resample.return_value = np.zeros((4, 2000), dtype=np.float32)

        input_path = "/tmp/fake_input/my_track.wav"
        result_path = process_audio(input_path)

        mock_load.assert_called_once_with(input_path, sr=None)
        mock_resample.assert_called_once()
        self.assertEqual(mock_resample.call_args.kwargs["target_sr"], 44100)

        written_path, written_y = (
            mock_write.call_args[0][0],
            mock_write.call_args.kwargs["y"],
        )
        self.assertEqual(written_y.shape, (2, 2000))
        self.assertEqual(mock_write.call_args.kwargs["samplerate"], 44100)

        expected_dir = _expected_processed_dir()
        self.assertEqual(os.path.dirname(written_path), expected_dir)
        self.assertEqual(os.path.basename(written_path), "my_track_processed.wav")
        self.assertEqual(result_path, written_path)


if __name__ == "__main__":
    unittest.main()
