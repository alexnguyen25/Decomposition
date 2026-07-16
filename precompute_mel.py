"""Throwaway script to precompute mel spectrograms for OpenMIC-2018."""

import numpy as np

from src.classification.dataset import OpenMICDataset
from src.config import OPENMIC_DIR, CACHE_DIR
from src.feature_extraction.feature_extraction import extract_mel_spectrogram

PARTITIONS = ("split01_train.csv", "split01_test.csv")


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0

    for partition in PARTITIONS:
        dataset = OpenMICDataset(OPENMIC_DIR, partition)
        n = len(dataset)
        print(f"{partition}: {n} clips")

        for i in range(n):
            key = str(dataset.sample_key[i])
            cache_path = CACHE_DIR / f"{key}.npy"

            if cache_path.exists():
                skipped += 1
            else:
                audio_path = OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg"
                mel = extract_mel_spectrogram(audio_path)
                np.save(cache_path, mel.astype(np.float32))
                saved += 1

            done = i + 1
            if done % 100 == 0 or done == n:
                print(f"  {partition}: {done}/{n} ({saved} saved, {skipped} skipped)")

    print(f"Done. {saved} new spectrograms written, {skipped} already cached.")


if __name__ == "__main__":
    main()
