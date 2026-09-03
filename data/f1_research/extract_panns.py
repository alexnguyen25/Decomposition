"""E7a: extract PANNs CNN14 (AudioSet-pretrained) 2048-d embeddings for all
OpenMIC clips. Decoding on CPU workers, model on MPS (CPU fallback).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH / "pylibs"))
PROJECT = Path("/Users/alexnguyen25/Documents/GitHub/Decomposition")
OPENMIC_DIR = PROJECT / "data" / "openmic" / "openmic-2018"

import librosa  # noqa: E402
from panns_inference.models import Cnn14  # noqa: E402

SR32 = 32000
N_SAMPLES = SR32 * 10  # 10s clips


class ClipDataset(Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        path = OPENMIC_DIR / "audio" / key[:3] / f"{key}.ogg"
        y, _ = librosa.load(path, sr=SR32, mono=True)
        out = np.zeros(N_SAMPLES, dtype=np.float32)
        n = min(len(y), N_SAMPLES)
        out[:n] = y[:n]
        return out


def keys_for(partition):
    with open(OPENMIC_DIR / "partitions" / partition) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                  mel_bins=64, fmin=50, fmax=14000, classes_num=527)
    ckpt = torch.load(SCRATCH / "Cnn14_mAP=0.431.pth", map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = model.to(device)
        with torch.no_grad():
            model(torch.zeros(2, N_SAMPLES, device=device))
        print(f"using {device}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"MPS failed ({e}); falling back to CPU", flush=True)
        device = torch.device("cpu")
        model = model.to(device)

    for partition in ("split01_test.csv", "split01_train.csv"):
        keys = keys_for(partition)
        out_path = SCRATCH / f"panns_{partition.replace('.csv', '')}.npz"
        if out_path.exists():
            print(f"{partition}: already extracted", flush=True)
            continue
        ds = ClipDataset(keys)
        dl = DataLoader(ds, batch_size=16, num_workers=6, shuffle=False)
        embs = []
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                out = model(batch.to(device))
                embs.append(out["embedding"].cpu().numpy())
                if (bi + 1) % 50 == 0:
                    print(f"  {partition}: {(bi + 1) * 16}/{len(keys)}",
                          flush=True)
        embs = np.concatenate(embs).astype(np.float32)
        np.savez(out_path, embeddings=embs, keys=np.array(keys))
        print(f"{partition}: saved {embs.shape}", flush=True)
    print("PANNS EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
