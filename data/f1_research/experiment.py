"""F1-improvement experiments for the Decomposition OpenMIC classifier.

Scratchpad research code — does not touch src/. Loads the same cached mels,
labels, and masked-F1 metric as the project, then trains/evaluates variants.

Usage: experiment.py E1 E2 E3   (runs sequentially, appends to results.jsonl)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT = Path("/Users/alexnguyen25/Documents/GitHub/Decomposition")
SCRATCH = Path(__file__).parent
OPENMIC_DIR = PROJECT / "data" / "openmic" / "openmic-2018"
CACHE_DIR = PROJECT / "data" / "openmic" / "mel_cache"
RESULTS = SCRATCH / "results.jsonl"

SEED = 42
VAL_FRACTION = 0.15
N_MELS, T_FRAMES = 128, 431

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------- data

def load_partition(partition: str):
    """Return (X float16 (N,128,431), Y float32 (N,20), M float32 (N,20), keys)."""
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    y_true, y_mask, sample_key = npz["Y_true"], npz["Y_mask"], npz["sample_key"]

    with open(OPENMIC_DIR / "partitions" / partition) as f:
        keys = [line.strip() for line in f if line.strip()]
    keyset = set(keys)
    sel = np.array([k in keyset for k in sample_key])

    y = (y_true[sel] >= 0.5).astype(np.float32)
    m = y_mask[sel].astype(np.float32)
    kept_keys = sample_key[sel]

    x = np.full((sel.sum(), N_MELS, T_FRAMES), -80.0, dtype=np.float16)
    for i, key in enumerate(kept_keys):
        mel = np.load(CACHE_DIR / f"{key}.npy")
        w = min(mel.shape[1], T_FRAMES)
        x[i, :, :w] = mel[:, :w].astype(np.float16)
    return x, y, m, kept_keys


def train_val_split(n: int):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(n)
    n_val = int(n * VAL_FRACTION)
    return idx[n_val:], idx[:n_val]


# ---------------------------------------------------------------- models

class BaselineModel(nn.Module):
    """Exact copy of src/classification/model.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 20)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.fc(self.gap(x).squeeze(dim=(2, 3)))


def bn_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
        nn.MaxPool2d(2),
    )


class CNN4(nn.Module):
    """4 conv blocks with BatchNorm, GAP head with dropout."""

    def __init__(self, chs=(32, 64, 128, 256), p_drop=0.3):
        super().__init__()
        blocks, cin = [], 1
        for c in chs:
            blocks.append(bn_block(cin, c))
            cin = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(p_drop),
            nn.Linear(cin, 20),
        )

    def forward(self, x):
        return self.head(self.features(x))


# ---------------------------------------------------------------- metric
# Mirrors src/classification/evaluate.py::compute_metrics (masked per-class F1).

def masked_f1(probs: np.ndarray, labels: np.ndarray, masks: np.ndarray,
              thresholds) -> dict:
    thresholds = np.broadcast_to(np.asarray(thresholds, dtype=np.float32), (20,))
    preds = (probs >= thresholds[None, :]).astype(np.int8)
    per_class = []
    for c in range(20):
        conf = masks[:, c] == 1
        yp, yt = preds[conf, c], labels[conf, c]
        tp = int(((yp == 1) & (yt == 1)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per_class.append({"class_index": c, "precision": p, "recall": r, "f1": f1})
    return {"classes": per_class,
            "macro_f1": float(np.mean([c["f1"] for c in per_class]))}


def tune_thresholds(probs, labels, masks, grid=None):
    """Per-class threshold maximizing F1 on the given (validation) set."""
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.025)
    best = np.full(20, 0.5)
    for c in range(20):
        conf = masks[:, c] == 1
        pc, yc = probs[conf, c], labels[conf, c]
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


# ---------------------------------------------------------------- training

def spec_augment(x: torch.Tensor, n_freq=2, freq_w=16, n_time=2, time_w=40):
    """In-place-ish masking on a (B,1,128,431) batch. Mask value 0 = padded floor."""
    b = x.shape[0]
    fill = x.min()
    for _ in range(n_freq):
        f0 = torch.randint(0, N_MELS - freq_w, (b,))
        w = torch.randint(1, freq_w + 1, (b,))
        for i in range(b):
            x[i, :, f0[i]:f0[i] + w[i], :] = fill
    for _ in range(n_time):
        t0 = torch.randint(0, T_FRAMES - time_w, (b,))
        w = torch.randint(1, time_w + 1, (b,))
        for i in range(b):
            x[i, :, :, t0[i]:t0[i] + w[i]] = fill
    return x


def predict(model, x_np, normalize, batch_size=128):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x_np), batch_size):
            xb = torch.from_numpy(x_np[s:s + batch_size].astype(np.float32))
            xb = xb.unsqueeze(1).to(DEVICE)
            if normalize:
                xb = (xb + 40.0) / 40.0
            outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs)


def compute_pos_weight(y, m, cap=8.0):
    """Per-class neg/pos ratio among confirmed labels, capped."""
    pos = (y * m).sum(axis=0)
    neg = ((1 - y) * m).sum(axis=0)
    w = np.clip(neg / np.maximum(pos, 1.0), 1.0, cap)
    return torch.from_numpy(w.astype(np.float32))


def train_model(model, data, cfg):
    x_tr, y_tr, m_tr, x_val, y_val, m_val = data
    model = model.to(DEVICE)
    opt = (torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
           if cfg["opt"] == "adamw"
           else torch.optim.Adam(model.parameters(), lr=cfg["lr"]))
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["max_epochs"])
             if cfg["cosine"] else None)
    pw = (compute_pos_weight(y_tr, m_tr).to(DEVICE)
          if cfg.get("use_pos_weight") else None)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)

    y_tr_t = torch.from_numpy(y_tr)
    m_tr_t = torch.from_numpy(m_tr)
    n = len(x_tr)
    best_f1, best_state, best_epoch, since_best = -1.0, None, -1, 0
    history = []

    for epoch in range(cfg["max_epochs"]):
        model.train()
        perm = np.random.permutation(n)
        total, nb = 0.0, 0
        for s in range(0, n, cfg["batch"]):
            sel = perm[s:s + cfg["batch"]]
            xb = torch.from_numpy(x_tr[sel].astype(np.float32)).unsqueeze(1)
            if cfg["normalize"]:
                xb = (xb + 40.0) / 40.0
            if cfg["specaug"]:
                xb = spec_augment(xb)
            xb = xb.to(DEVICE)
            yb, mb = y_tr_t[sel].to(DEVICE), m_tr_t[sel].to(DEVICE)

            opt.zero_grad()
            per_el = loss_fn(model(xb), yb)
            loss = (per_el * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        if sched:
            sched.step()

        val_probs = predict(model, x_val, cfg["normalize"])
        val_f1 = masked_f1(val_probs, y_val, m_val, 0.5)["macro_f1"]
        history.append({"epoch": epoch, "train_loss": total / nb, "val_f1": val_f1})
        print(f"  epoch {epoch:02d} loss {total / nb:.4f} val_f1 {val_f1:.4f}",
              flush=True)

        if val_f1 > best_f1:
            best_f1, best_epoch, since_best = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since_best += 1
            if since_best >= cfg["patience"]:
                print(f"  early stop at epoch {epoch} (best {best_epoch})",
                      flush=True)
                break

    model.load_state_dict(best_state)
    return model, {"best_val_f1": best_f1, "best_epoch": best_epoch,
                   "history": history}


# ---------------------------------------------------------------- experiments

RECIPE = dict(opt="adamw", lr=1e-3, wd=1e-4, cosine=True, batch=64,
              max_epochs=40, patience=6, normalize=True, specaug=False)

EXPERIMENTS = {
    # recipe-only on the existing tiny architecture
    "E1_tiny_recipe": dict(model=lambda: BaselineModel(), **RECIPE),
    # deeper CNN with batchnorm
    "E2_cnn4": dict(model=lambda: CNN4(), **{**RECIPE, "lr": 3e-4}),
    # + SpecAugment
    "E3_cnn4_specaug": dict(model=lambda: CNN4(),
                            **{**RECIPE, "lr": 3e-4, "specaug": True,
                               "max_epochs": 50, "patience": 8}),
    # + pos_weight for class imbalance
    "E4_cnn4_specaug_posweight": dict(model=lambda: CNN4(),
                                      **{**RECIPE, "lr": 3e-4, "specaug": True,
                                         "max_epochs": 50, "patience": 8,
                                         "use_pos_weight": True}),
}


def append_result(record):
    with open(RESULTS, "a") as f:
        f.write(json.dumps(record) + "\n")


def main(names):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading train partition...", flush=True)
    x_all, y_all, m_all, _ = load_partition("split01_train.csv")
    tr_idx, val_idx = train_val_split(len(x_all))
    data = (x_all[tr_idx], y_all[tr_idx], m_all[tr_idx],
            x_all[val_idx], y_all[val_idx], m_all[val_idx])
    print(f"train {len(tr_idx)} / val {len(val_idx)}", flush=True)

    print("loading test partition...", flush=True)
    x_te, y_te, m_te, _ = load_partition("split01_test.csv")

    for name in names:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        cfg = dict(EXPERIMENTS[name])
        model_fn = cfg.pop("model")
        print(f"\n=== {name} ===", flush=True)
        t0 = time.time()
        model, train_info = train_model(model_fn(), data, cfg)
        minutes = (time.time() - t0) / 60

        test_probs = predict(model, x_te, cfg["normalize"])
        m_default = masked_f1(test_probs, y_te, m_te, 0.5)

        val_probs = predict(model, data[3], cfg["normalize"])
        ths = tune_thresholds(val_probs, data[4], data[5])
        m_tuned = masked_f1(test_probs, y_te, m_te, ths)

        torch.save(model.state_dict(), SCRATCH / f"ckpt_{name}.pt")
        record = {
            "name": name, "minutes": round(minutes, 1),
            "best_val_f1": train_info["best_val_f1"],
            "best_epoch": train_info["best_epoch"],
            "test_macro_f1_at_0.5": m_default["macro_f1"],
            "test_macro_f1_tuned_th": m_tuned["macro_f1"],
            "tuned_thresholds": ths.tolist(),
            "per_class_f1_at_0.5": [c["f1"] for c in m_default["classes"]],
            "per_class_f1_tuned": [c["f1"] for c in m_tuned["classes"]],
            "history": train_info["history"],
        }
        append_result(record)
        print(f"{name}: test macro-F1 @0.5 = {m_default['macro_f1']:.4f}, "
              f"tuned = {m_tuned['macro_f1']:.4f} ({minutes:.0f} min)", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or list(EXPERIMENTS))
