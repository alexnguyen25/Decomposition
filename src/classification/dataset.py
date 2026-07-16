"""
PyTorch dataset for OpenMIC-2018 instrument classification.

Loads crowd-sourced multi-label annotations from ``openmic-2018.npz`` and
filters rows to a train/test partition. Each sample returns a mel spectrogram
computed with the same ``extract_mel_spectrogram`` function used at inference.

Expected directory layout (after extracting the Zenodo archive)::

    openmic_dir/
    ├── audio/{key[:3]}/{key}.ogg
    ├── openmic-2018.npz          # Y_true, Y_mask, sample_key
    └── partitions/
        ├── train01.txt           # one sample key per line
        └── test01.txt

Labels are weak: ``Y_true`` holds confidence scores in [0, 1] and ``Y_mask``
marks which entries are confirmed. Training should threshold labels at 0.5 and
multiply loss by ``Y_mask`` so unconfirmed classes are ignored.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.feature_extraction.feature_extraction import extract_mel_spectrogram


class OpenMICDataset(Dataset):
    """OpenMIC-2018 clips as mel spectrograms with multi-hot instrument labels."""

    def __init__(
        self,
        openmic_dir: Path | str,
        partition: str,
        cache_dir: Path | str | None = None,
    ) -> None:
        """
        Load annotations and restrict samples to a partition file.

        Args:
            openmic_dir: Root of the extracted OpenMIC-2018 tree.
            partition: Partition filename (e.g. ``"train01.txt"``) or stem
                without extension (e.g. ``"train01"`` → ``train01.txt``).
            cache_dir: Optional directory of precomputed ``{sample_key}.npy``
                mel spectrograms. When a cache file exists, audio is not decoded.

        Attributes:
            Y_true: Filtered label confidences, shape ``(n, 20)``.
            Y_mask: Filtered confirmation mask, shape ``(n, 20)``.
            sample_key: Filtered clip identifiers, shape ``(n,)``.
        """
        self.openmic_dir = Path(openmic_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        npz = np.load(self.openmic_dir / "openmic-2018.npz", allow_pickle=True)
        y_true = npz["Y_true"]
        y_mask = npz["Y_mask"]
        sample_key = npz["sample_key"]

        partition_file = partition if "." in partition else f"{partition}.txt"
        partition_path = self.openmic_dir / "partitions" / partition_file
        with open(partition_path) as f:
            partition_keys = {line.strip() for line in f if line.strip()}

        mask = np.isin(sample_key, list(partition_keys))
        self.Y_true = y_true[mask]
        self.Y_mask = y_mask[mask]
        self.sample_key = sample_key[mask]

    def __len__(self) -> int:
        """Return the number of clips in this partition."""
        return len(self.sample_key)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load one clip and return spectrogram, binary labels, and label mask.

        Args:
            i: Index into the filtered partition.

        Returns:
            A tuple of:
            - ``spec``: Mel spectrogram, shape ``(1, n_mels, t)`` (channel first).
            - ``labels``: Binary multi-label vector, shape ``(20,)`` (threshold 0.5).
            - ``label_mask``: Float mask for loss weighting, shape ``(20,)``.
        """
        key = str(self.sample_key[i])
        cache_path = self.cache_dir / f"{key}.npy" if self.cache_dir else None

        if cache_path is not None and cache_path.exists():
            mel = np.load(cache_path)
        else:
            audio_path = self.openmic_dir / "audio" / key[:3] / f"{key}.ogg"
            mel = extract_mel_spectrogram(audio_path)

        spec = torch.from_numpy(mel).unsqueeze(0).float()

        labels = torch.from_numpy((self.Y_true[i] >= 0.5).astype(np.float32))
        label_mask = torch.from_numpy(self.Y_mask[i].astype(np.float32))

        return spec, labels, label_mask
