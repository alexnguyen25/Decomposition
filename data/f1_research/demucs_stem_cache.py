"""Build a stem-domain dataset: Demucs "other"-stem 32k audio for
(a) 300 train-partition clips (calibration/fine-tune set) and
(b) the same 80 test clips used in the shift study (evaluation set).

Saves float16 waveforms to stem_audio_cache/{key}.npy. 3 worker processes.
"""

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))

OUT = SCRATCH / "stem_audio_cache"
N_TRAIN = 300
SEED_TRAIN = 43


def train_keys():
    from experiment import OPENMIC_DIR
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    y_mask, sample_key = npz["Y_mask"], npz["sample_key"]
    with open(OPENMIC_DIR / "partitions" / "split01_train.csv") as f:
        keys = {line.strip() for line in f if line.strip()}
    sel = np.array([str(k) in keys for k in sample_key])
    kept = sample_key[sel]
    has_label = y_mask[sel].sum(axis=1) > 0
    idx = np.flatnonzero(has_label)
    rng = np.random.default_rng(SEED_TRAIN)
    chosen = rng.choice(idx, size=N_TRAIN, replace=False)
    return [str(k) for k in kept[chosen]]


def test_keys():
    from demucs_shift import sample_clips
    keys, _, _ = sample_clips()
    return keys


def separate_one(key: str) -> str:
    out_path = OUT / f"{key}.npy"
    if out_path.exists():
        return "skip"
    import torch
    torch.set_num_threads(3)
    import librosa
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    global _DM
    try:
        _DM
    except NameError:
        _DM = get_model("htdemucs")
        _DM.eval()
    from experiment import OPENMIC_DIR
    audio = OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg"
    wav, _ = librosa.load(audio, sr=44100, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    with torch.no_grad():
        sources = apply_model(_DM, torch.from_numpy(wav).float().unsqueeze(0),
                              device="cpu", progress=False)[0]
    other = sources[_DM.sources.index("other")].mean(0).numpy()
    other32 = librosa.resample(other, orig_sr=44100, target_sr=32000)
    tmp = out_path.with_suffix(".tmp.npy")
    np.save(tmp, other32.astype(np.float16))
    tmp.rename(out_path)
    return "saved"


def main():
    OUT.mkdir(exist_ok=True)
    keys = train_keys() + test_keys()
    print(f"{len(keys)} clips to separate", flush=True)
    done = 0
    with Pool(3) as pool:
        for _ in pool.imap_unordered(separate_one, keys, chunksize=4):
            done += 1
            if done % 20 == 0 or done == len(keys):
                print(f"  {done}/{len(keys)}", flush=True)
    print("STEM CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
