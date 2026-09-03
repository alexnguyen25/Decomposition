"""E10a: extract frozen BEATs (iter3+ AS2M) embeddings for all OpenMIC clips.

Protocol per Quelennec et al. 2024 (OpenMIC frozen SOTA): 5s windows; we use
two 5s windows per 10s clip, mean token pooling, mean over windows -> 768-d.
Kaldi fbank runs on CPU (op unsupported on MPS); encoder on MPS if possible.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "beats"))
from experiment import OPENMIC_DIR  # noqa: E402
from BEATs import BEATs, BEATsConfig  # noqa: E402

import librosa  # noqa: E402

SR16 = 16000
WIN = SR16 * 5


class Clips(Dataset):
    def __init__(self, keys):
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        k = self.keys[i]
        y, _ = librosa.load(OPENMIC_DIR / "audio" / k[:3] / f"{k}.ogg",
                            sr=SR16, mono=True)
        out = np.zeros(SR16 * 10, dtype=np.float32)
        out[:min(len(y), SR16 * 10)] = y[:SR16 * 10]
        return k, out


class Encoder:
    def __init__(self):
        ckpt = torch.load(SCRATCH / "beats" / "BEATs_iter3_plus_AS2M.pt",
                          map_location="cpu")
        cfg = BEATsConfig(ckpt["cfg"])
        self.model = BEATs(cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.device = torch.device("cpu")
        if torch.backends.mps.is_available():
            try:
                self.model.to("mps")
                self._encode(torch.zeros(2, 498, 128, device="mps"))
                self.device = torch.device("mps")
                print("encoder on MPS", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"MPS failed ({str(e)[:120]}); CPU fallback", flush=True)
                self.model.to("cpu")

    def _encode(self, fbank):
        m = self.model
        x = fbank.unsqueeze(1)
        x = m.patch_embedding(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = m.layer_norm(x)
        if m.post_extract_proj is not None:
            x = m.post_extract_proj(x)
        x = m.dropout_input(x)
        x, _ = m.encoder(x, padding_mask=None)
        return x  # (B, T', 768)

    def embed_batch(self, waves):  # (B, 160000)
        b = waves.shape[0]
        windows = torch.cat([waves[:, :WIN], waves[:, WIN:]], dim=0)  # (2B, WIN)
        with torch.no_grad():
            fbank = self.model.preprocess(windows.cpu())  # kaldi: CPU only
            tokens = self._encode(fbank.to(self.device))
            emb = tokens.mean(dim=1)                      # (2B, 768)
            emb = (emb[:b] + emb[b:]) / 2                 # avg windows
        return emb.cpu().numpy()


def keys_for(partition):
    with open(OPENMIC_DIR / "partitions" / partition) as f:
        return [l.strip() for l in f if l.strip()]


def main():
    enc = Encoder()

    # timing probe
    probe = torch.randn(8, SR16 * 10)
    t0 = time.time()
    enc.embed_batch(probe)
    per8 = time.time() - t0
    print(f"probe: {per8:.2f}s per 8 clips -> est "
          f"{per8 / 8 * 20000 / 60:.0f} min for 20k", flush=True)

    for partition in ("split01_test.csv", "split01_train.csv"):
        keys = keys_for(partition)
        out_path = SCRATCH / f"beats_{partition.replace('.csv', '')}.npz"
        if out_path.exists():
            print(f"{partition}: already extracted", flush=True)
            continue
        dl = DataLoader(Clips(keys), batch_size=16, num_workers=6,
                        shuffle=False)
        embs = []
        done = 0
        for ks, waves in dl:
            embs.append(enc.embed_batch(waves))
            done += len(ks)
            if done % 1600 < 16:
                print(f"  {partition}: {done}/{len(keys)}", flush=True)
        np.savez(out_path, embeddings=np.concatenate(embs).astype(np.float32),
                 keys=np.array(keys))
        print(f"{partition}: saved", flush=True)
    print("BEATS EXTRACT DONE", flush=True)


if __name__ == "__main__":
    main()
