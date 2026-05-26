"""
Unit tests for ``src.separation.separation``.

Covers:
    - ``_collect_stems``: pure path construction (no mocks).
    - ``_run_demucs``: subprocess invocation and error mapping.
    - ``separate``: orchestration logic with internal helpers patched.

Run:
    python3 -m pytest tests/separation/test_separation.py -v
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from src.separation.separation import _collect_stems, _run_demucs, separate
from src.utils.exceptions import DemucsFail, DemucsNotFound


class TestCollectStems(unittest.TestCase):
    def test_collect_stems_constructs_expected_paths(self):
        output_dir = Path("/tmp/out")
        model = "htdemucs"
        track_name = "my_track"

        result = _collect_stems(output_dir, track_name, model)

        self.assertEqual(
            result,
            {
                "vocals": output_dir / model / track_name / "vocals.wav",
                "drums": output_dir / model / track_name / "drums.wav",
                "bass": output_dir / model / track_name / "bass.wav",
                "other": output_dir / model / track_name / "other.wav",
            },
        )


class TestRunDemucs(unittest.TestCase):
    @patch("src.separation.separation.subprocess.run")
    def test_run_demucs_success_calls_subprocess_with_expected_command(self, mock_run):
        file_path = Path("/tmp/in/my_track.wav")
        output_dir = Path("/tmp/out")
        model = "htdemucs"

        _run_demucs(file_path, output_dir, model)

        mock_run.assert_called_once_with(
            [
                "python3",
                "-m",
                "demucs.separate",
                "-n",
                model,
                "-o",
                str(output_dir),
                str(file_path),
            ],
            check=True,
            capture_output=True,
        )

    @patch("src.separation.separation.subprocess.run")
    def test_run_demucs_called_process_error_raises_demucs_fail(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=["python3", "-m", "demucs.separate"]
        )

        with self.assertRaises(DemucsFail):
            _run_demucs(Path("/tmp/in/bad.wav"), Path("/tmp/out"), "htdemucs")

    @patch("src.separation.separation.subprocess.run")
    def test_run_demucs_file_not_found_raises_demucs_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()

        with self.assertRaises(DemucsNotFound):
            _run_demucs(Path("/tmp/in/a.wav"), Path("/tmp/out"), "htdemucs")


class TestSeparate(unittest.TestCase):
    @patch("src.separation.separation._collect_stems")
    @patch("src.separation.separation._run_demucs")
    def test_separate_calls_helpers_and_uses_derived_track_name(
        self, mock_run_demucs, mock_collect_stems
    ):
        file_path = Path("/tmp/in/track.wav")
        output_dir = Path("/tmp/out")
        model = "htdemucs"

        expected = {"vocals": Path("/tmp/out/htdemucs/track/vocals.wav")}
        mock_collect_stems.return_value = expected

        result = separate(file_path, output_dir, model=model)

        mock_run_demucs.assert_called_once_with(file_path, output_dir, model)
        mock_collect_stems.assert_called_once_with(output_dir, file_path.stem, model)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()

