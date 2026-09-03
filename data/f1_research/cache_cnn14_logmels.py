"""Cache CNN14-format logmels (T~1001, 64) f16 for all OpenMIC clips.
Decode on 6 CPU workers, frontend (STFT+mel) on MPS. Enables fast fine-tuning
without re-decoding audio every epoch.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "pylibs"))
from experiment import OPENMIC_DIR  # noqa: E402
from panns_inference.models import Cnn14  # noqa: E402

import librosa  # noqa: E402

OUT = SCRATCH / "cnn14_logmel_cache"
SR32, N = 32000, 320000
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Clips(Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        y, _ = librosa.load(OPENMIC_DIR / "audio" / k[:3] / f"{k}.ogg",
                            sr=SR32, mono=True)
        out = np.zeros(N, dtype=np.float32)
        out[:min(len(y), N)] = y[:N]
        return k, out


def main():
    OUT.mkdir(exist_ok=True)
    keys = []
    for part in ("split01_train.csv", "split01_test.csv"):
        with open(OPENMIC_DIR / "partitions" / part) as f:
            keys += [l.strip() for l in f if l.strip()]
    keys = [k for k in keys if not (OUT / f"{k}.npy").exists()]
    print(f"{len(keys)} logmels to compute", flush=True)
    if not keys:
        print("LOGMEL CACHE DONE", flush=True)
        return

    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    front = torch.nn.Sequential().to(DEVICE)  # placeholder
    spec, logmel = m.spectrogram_extractor.to(DEVICE), \
        m.logmel_extractor.to(DEVICE)

    dl = DataLoader(Clips(keys), batch_size=32, num_workers=6)
    done = 0
    with torch.no_grad():
        for ks, waves in dl:
            x = logmel(spec(waves.to(DEVICE))).squeeze(1).cpu().numpy()
            for j, k in enumerate(ks):
                np.save(OUT / f"{k}.npy", x[j].astype(np.float16))
            done += len(ks)
            if done % 1600 < 32:
                print(f"  {done}/{len(keys)}", flush=True)
    print("LOGMEL CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
