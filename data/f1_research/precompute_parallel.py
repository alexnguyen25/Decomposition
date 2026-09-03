"""Parallel mel-spectrogram precompute for OpenMIC-2018 (scratchpad tool).

Produces cache files identical to the project's precompute_mel.py by importing
the project's own extract_mel_spectrogram. Test partition first so eval can
run before the full cache finishes.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT = Path("/Users/alexnguyen25/Documents/GitHub/Decomposition")
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402

OPENMIC_DIR = PROJECT / "data" / "openmic" / "openmic-2018"
CACHE_DIR = PROJECT / "data" / "openmic" / "mel_cache"
PARTITIONS = ("split01_test.csv", "split01_train.csv")


def keys_for_partition(partition: str) -> list[str]:
    path = OPENMIC_DIR / "partitions" / partition
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def compute_one(key: str) -> str:
    cache_path = CACHE_DIR / f"{key}.npy"
    if cache_path.exists():
        return "skip"
    from src.feature_extraction.feature_extraction import extract_mel_spectrogram

    audio_path = OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg"
    mel = extract_mel_spectrogram(audio_path)
    tmp = cache_path.with_suffix(".tmp.npy")
    np.save(tmp, mel.astype(np.float32))
    tmp.rename(cache_path)
    return "saved"


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for partition in PARTITIONS:
        keys = keys_for_partition(partition)
        print(f"{partition}: {len(keys)} clips", flush=True)
        done = 0
        with Pool(8) as pool:
            for _ in pool.imap_unordered(compute_one, keys, chunksize=32):
                done += 1
                if done % 500 == 0 or done == len(keys):
                    print(f"  {partition}: {done}/{len(keys)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
