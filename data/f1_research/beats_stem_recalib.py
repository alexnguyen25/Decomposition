"""Train the BEATs stem-domain head variant (recalib-B recipe on BEATs).

Embeds the 380 cached Demucs 'other' stems with BEATs, fine-tunes the E10
head on full-mix train embeddings + 3x-oversampled stem embeddings, evaluates
on the 80 stem-domain test clips + Jamendo recall, saves
ckpt_E10_stem_recalib.pt.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "beats"))
from experiment import DEVICE, OPENMIC_DIR  # noqa: E402
from beats_head import MLPHead  # noqa: E402
from stem_recalib import (  # noqa: E402
    KEEP, kept_f1_fp, labels_for, tune_th,
)
from BEATs import BEATs, BEATsConfig  # noqa: E402

STEMS = SCRATCH / "stem_audio_cache"
SR16, WIN5 = 16000, 16000 * 5


class BeatsEmbedder:
    def __init__(self):
        ckpt = torch.load(SCRATCH / "beats" / "BEATs_iter3_plus_AS2M.pt",
                          map_location="cpu")
        self.m = BEATs(BEATsConfig(ckpt["cfg"]))
        self.m.load_state_dict(ckpt["model"])
        self.m.eval().to(DEVICE)

    def _encode(self, fbank):
        m = self.m
        x = fbank.unsqueeze(1)
        x = m.patch_embedding(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        x = m.layer_norm(x)
        if m.post_extract_proj is not None:
            x = m.post_extract_proj(x)
        x, _ = m.encoder(m.dropout_input(x), padding_mask=None)
        return x

    def embed_10s(self, waves_16k):  # (B, 160000) numpy
        t = torch.from_numpy(waves_16k.astype(np.float32))
        b = t.shape[0]
        wins = torch.cat([t[:, :WIN5], t[:, WIN5:WIN5 * 2]], dim=0)
        with torch.no_grad():
            fb = self.m.preprocess(wins.cpu())
            tok = self._encode(fb.to(DEVICE))
            emb = tok.mean(1)
            return ((emb[:b] + emb[b:]) / 2).cpu().numpy()


def stem_wave_16k(key):
    import librosa
    w = np.load(STEMS / f"{key}.npy").astype(np.float32)  # 32k
    w16 = librosa.resample(w, orig_sr=32000, target_sr=SR16)
    out = np.zeros(SR16 * 10, dtype=np.float32)
    out[:min(len(w16), SR16 * 10)] = w16[:SR16 * 10]
    return out


def main():
    emb = BeatsEmbedder()
    all_keys = sorted(p.stem for p in STEMS.glob("*.npy"))
    with open(OPENMIC_DIR / "partitions" / "split01_test.csv") as f:
        test_part = {l.strip() for l in f if l.strip()}
    eval_keys = [k for k in all_keys if k in test_part]
    cal_keys = [k for k in all_keys if k not in test_part]
    print(f"cal {len(cal_keys)} / eval {len(eval_keys)}", flush=True)

    def embed_keys(keys):
        out = np.zeros((len(keys), 768), dtype=np.float32)
        for s in range(0, len(keys), 16):
            batch = np.stack([stem_wave_16k(k) for k in keys[s:s + 16]])
            out[s:s + 16] = emb.embed_10s(batch)
        return out

    cal_x, eval_x = embed_keys(cal_keys), embed_keys(eval_keys)
    cal_y, cal_m = labels_for(cal_keys)
    eval_y, eval_m = labels_for(eval_keys)
    np.savez(SCRATCH / "beats_stem_embeddings.npz",
             cal_x=cal_x, eval_x=eval_x,
             cal_keys=np.array(cal_keys), eval_keys=np.array(eval_keys))

    head = MLPHead().to(DEVICE)
    head.load_state_dict(torch.load(SCRATCH / "ckpt_E10_beats_head.pt",
                                    map_location=DEVICE, weights_only=True))
    head.eval()

    def probs(h, x):
        with torch.no_grad():
            return torch.sigmoid(
                h(torch.from_numpy(x).to(DEVICE))).cpu().numpy()

    f1, fp = kept_f1_fp(probs(head, eval_x), eval_y, eval_m, 0.5)
    print(f"E10 baseline on stem-domain eval: kept-F1 {f1:.4f}, FPs {fp}",
          flush=True)

    # recalib-B recipe: mix embeddings + 3x stems, 4 epochs lr 2e-4
    mix = np.load(SCRATCH / "beats_split01_train.npz", allow_pickle=True)
    mix_keys = [str(k) for k in mix["keys"]]
    mix_y, mix_m = labels_for(mix_keys)
    x = np.concatenate([mix["embeddings"]] + [cal_x] * 3)
    y = np.concatenate([mix_y] + [cal_y] * 3)
    m = np.concatenate([mix_m] + [cal_m] * 3)
    xt, yt, mt = (torch.from_numpy(a) for a in (x, y, m))
    opt = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
    n, batch = len(x), 256
    for epoch in range(4):
        head.train()
        perm = torch.randperm(n)
        tot = nb = 0
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            xb, yb, mb = (t[sel].to(DEVICE) for t in (xt, yt, mt))
            opt.zero_grad()
            per = loss_fn(head(xb), yb)
            loss = (per * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        print(f"  epoch {epoch} loss {tot / nb:.4f}", flush=True)
    head.eval()

    f1b, fpb = kept_f1_fp(probs(head, eval_x), eval_y, eval_m, 0.5)
    print(f"E10 stem-recalib on stem-domain eval: kept-F1 {f1b:.4f}, "
          f"FPs {fpb}", flush=True)

    # jamendo real-song check
    with open(SCRATCH / "jamendo" / "manifest.json") as f:
        manifest = json.load(f)
    with open(OPENMIC_DIR / "class-map.json") as f:
        cm = json.load(f)
    name_of = {i: n for n, i in cm.items()}
    import librosa
    for label, h in [("recalib", head)]:
        tot_tags = tot_hits = tot_extras = 0
        for tid, meta in manifest.items():
            w = np.load(SCRATCH / "jamendo" / f"{tid}_other_32k.npy").astype(
                np.float32)
            w16 = librosa.resample(w, orig_sr=32000, target_sr=SR16)
            win = SR16 * 10
            chunks = [np.pad(w16[s:s + win],
                             (0, max(0, win - len(w16[s:s + win]))))
                      for s in range(0, max(1, len(w16) - SR16 * 3), win)]
            ps = []
            for s in range(0, len(chunks), 16):
                ps.append(probs(h, emb.embed_10s(np.stack(chunks[s:s + 16]))))
            p = np.concatenate(ps)
            song = np.sort(p, axis=0)[-3:].mean(0)
            pred = {name_of[c] for c in KEEP if song[c] >= 0.5}
            truth = set(meta["classes"])
            tot_tags += len(truth)
            tot_hits += len(pred & truth)
            tot_extras += len(pred - truth)
        print(f"jamendo[{label}]: recall {tot_hits / tot_tags:.3f} "
              f"extras/track {tot_extras / len(manifest):.2f}", flush=True)

    torch.save(head.state_dict(), SCRATCH / "ckpt_E10_stem_recalib.pt")
    print("BEATS STEM RECALIB DONE", flush=True)


if __name__ == "__main__":
    main()
