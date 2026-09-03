"""Stem-domain recalibration of the E7 head (the real-song over-prediction fix).

Data: stem_audio_cache/ = Demucs "other" stems for 300 train clips
(calibration) + 80 test clips (evaluation, same keys as the shift study).

Measures on the 80 stem-domain eval clips (16 kept classes):
  1. E7 baseline @0.5 and @full-mix-tuned thresholds
  2. Recalib-A: thresholds tuned on the 300 stem-domain calibration clips
  3. Recalib-B: head fine-tuned on full-mix train embeddings + stem embeddings
Then re-runs the 14 Jamendo tracks with the best recalibrated config.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "pylibs"))
from experiment import DEVICE, OPENMIC_DIR  # noqa: E402
from panns_head import MLPHead  # noqa: E402
from panns_inference.models import Cnn14  # noqa: E402

STEMS = SCRATCH / "stem_audio_cache"
DEAD = {2, 5, 6, 19}
KEEP = [c for c in range(20) if c not in DEAD]
WIN = 320000


def load_cnn14():
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527)
    m.load_state_dict(torch.load(SCRATCH / "Cnn14_mAP=0.431.pth",
                                 map_location="cpu")["model"])
    return m.to(DEVICE).eval()


def labels_for(keys):
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    row = {str(k): i for i, k in enumerate(npz["sample_key"])}
    idx = [row[k] for k in keys]
    return ((npz["Y_true"][idx] >= 0.5).astype(np.float32),
            npz["Y_mask"][idx].astype(np.float32))


def embed_stems(cnn14, keys):
    """Clip stems are 10s -> exactly one window each."""
    embs = np.zeros((len(keys), 2048), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(keys), 16):
            batch = []
            for k in keys[s:s + 16]:
                w = np.load(STEMS / f"{k}.npy").astype(np.float32)
                w = np.pad(w, (0, max(0, WIN - len(w))))[:WIN]
                batch.append(w)
            x = torch.from_numpy(np.stack(batch)).to(DEVICE)
            embs[s:s + 16] = cnn14(x)["embedding"].cpu().numpy()
    return embs


def head_probs(head, embs):
    with torch.no_grad():
        return torch.sigmoid(
            head(torch.from_numpy(embs).to(DEVICE))).cpu().numpy()


def kept_f1_fp(probs, y, m, th):
    th = np.broadcast_to(np.asarray(th, np.float32), (20,))
    f1s, fp_tot = [], 0
    for c in KEEP:
        conf = m[:, c] == 1
        yp = (probs[conf, c] >= th[c]).astype(int)
        yt = y[conf, c]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        fp_tot += fp
    return float(np.mean(f1s)), fp_tot


def tune_th(probs, y, m):
    grid = np.arange(0.05, 0.96, 0.025)
    best = np.full(20, 0.5)
    for c in range(20):
        conf = m[:, c] == 1
        pc, yc = probs[conf, c], y[conf, c]
        best_f1 = -1.0
        for t in grid:
            yp = pc >= t
            tp = int((yp & (yc == 1)).sum())
            fp = int((yp & (yc == 0)).sum())
            fn = int((~yp & (yc == 1)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            if f1 > best_f1:
                best_f1, best[c] = f1, t
    return best


def finetune_head(mix_x, mix_y, mix_m, stem_x, stem_y, stem_m):
    """Continue training E7 head on full-mix + stem-domain data (3x weight)."""
    head = MLPHead().to(DEVICE)
    head.load_state_dict(torch.load(SCRATCH / "ckpt_E7_panns_head.pt",
                                    map_location=DEVICE, weights_only=True))
    x = np.concatenate([mix_x] + [stem_x] * 3)
    y = np.concatenate([mix_y] + [stem_y] * 3)
    m = np.concatenate([mix_m] + [stem_m] * 3)
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
        print(f"  recalib-B epoch {epoch} loss {tot / nb:.4f}", flush=True)
    head.eval()
    return head


def jamendo_eval(head, ths, label):
    cnn14 = jamendo_eval.cnn14
    with open(SCRATCH / "jamendo" / "manifest.json") as f:
        manifest = json.load(f)
    with open(OPENMIC_DIR / "class-map.json") as f:
        cm = json.load(f)
    name_of = {i: n for n, i in cm.items()}
    tot_tags = tot_hits = tot_extras = 0
    for tid, meta in manifest.items():
        wave = np.load(SCRATCH / "jamendo" / f"{tid}_other_32k.npy").astype(
            np.float32)
        chunks = [np.pad(wave[s:s + WIN],
                         (0, max(0, WIN - len(wave[s:s + WIN]))))
                  for s in range(0, max(1, len(wave) - 96000), WIN)]
        with torch.no_grad():
            emb = cnn14(torch.from_numpy(np.stack(chunks)).to(DEVICE))[
                "embedding"]
            probs = torch.sigmoid(head(emb)).cpu().numpy()
        song = np.sort(probs, axis=0)[-3:].mean(0)
        pred = {name_of[c] for c in KEEP if song[c] >= ths[c]}
        truth = set(meta["classes"])
        tot_tags += len(truth)
        tot_hits += len(pred & truth)
        tot_extras += len(pred - truth)
    rec = tot_hits / tot_tags
    ext = tot_extras / len(manifest)
    print(f"  jamendo[{label}]: recall {rec:.3f} extras/track {ext:.2f}",
          flush=True)
    return {"recall": rec, "extras_per_track": ext}


def main():
    t0 = time.time()
    all_keys = sorted(p.stem for p in STEMS.glob("*.npy"))
    with open(OPENMIC_DIR / "partitions" / "split01_test.csv") as f:
        test_part = {l.strip() for l in f if l.strip()}
    eval_keys = [k for k in all_keys if k in test_part]
    cal_keys = [k for k in all_keys if k not in test_part]
    print(f"calibration {len(cal_keys)} / eval {len(eval_keys)} stems",
          flush=True)

    cnn14 = load_cnn14()
    jamendo_eval.cnn14 = cnn14
    cal_x = embed_stems(cnn14, cal_keys)
    eval_x = embed_stems(cnn14, eval_keys)
    cal_y, cal_m = labels_for(cal_keys)
    eval_y, eval_m = labels_for(eval_keys)

    head = MLPHead().to(DEVICE)
    head.load_state_dict(torch.load(SCRATCH / "ckpt_E7_panns_head.pt",
                                    map_location=DEVICE, weights_only=True))
    head.eval()
    with open(SCRATCH / "results.jsonl") as f:
        recs = {json.loads(l)["name"]: json.loads(l) for l in f}
    th_mix = np.array(recs["E7_panns_head"]["tuned_thresholds"])

    out = {}
    ev = head_probs(head, eval_x)
    for label, th in [("baseline@0.5", 0.5), ("baseline@mix-tuned", th_mix)]:
        f1, fp = kept_f1_fp(ev, eval_y, eval_m, th)
        out[label] = {"kept_macro_f1": f1, "fp": fp}
        print(f"{label}: kept-F1 {f1:.4f}, FPs {fp}", flush=True)

    cal_p = head_probs(head, cal_x)
    th_stem = tune_th(cal_p, cal_y, cal_m)
    f1, fp = kept_f1_fp(ev, eval_y, eval_m, th_stem)
    out["recalibA_stem_thresholds"] = {"kept_macro_f1": f1, "fp": fp,
                                       "thresholds": th_stem.tolist()}
    print(f"recalib-A (stem thresholds): kept-F1 {f1:.4f}, FPs {fp}",
          flush=True)

    mix = np.load(SCRATCH / "panns_split01_train.npz", allow_pickle=True)
    mix_keys = [str(k) for k in mix["keys"]]
    mix_y, mix_m = labels_for(mix_keys)
    head_b = finetune_head(mix["embeddings"], mix_y, mix_m,
                           cal_x, cal_y, cal_m)
    ev_b = head_probs(head_b, eval_x)
    for label, th in [("recalibB@0.5", np.full(20, 0.5))]:
        f1, fp = kept_f1_fp(ev_b, eval_y, eval_m, th)
        out[label] = {"kept_macro_f1": f1, "fp": fp}
        print(f"{label}: kept-F1 {f1:.4f}, FPs {fp}", flush=True)
    cal_pb = head_probs(head_b, cal_x)
    th_b = tune_th(cal_pb, cal_y, cal_m)
    f1, fp = kept_f1_fp(ev_b, eval_y, eval_m, th_b)
    out["recalibB@stem-tuned"] = {"kept_macro_f1": f1, "fp": fp}
    print(f"recalibB@stem-tuned: kept-F1 {f1:.4f}, FPs {fp}", flush=True)

    torch.save(head_b.state_dict(), SCRATCH / "ckpt_E7_stem_recalib.pt")
    np.save(SCRATCH / "stem_thresholds.npy", th_b)

    out["jamendo"] = {
        "baseline_top3mean@0.5": jamendo_eval(head, np.full(20, .5),
                                              "baseline"),
        "recalibB_top3mean@stem-th": jamendo_eval(head_b, th_b, "recalibB"),
    }
    with open(SCRATCH / "stem_recalib_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"STEM RECALIB DONE ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
