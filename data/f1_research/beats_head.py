"""E10b: MLP head on frozen BEATs embeddings (768-d). Same protocol as E7."""

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (  # noqa: E402
    DEVICE, OPENMIC_DIR, SEED, append_result, masked_f1, train_val_split,
    tune_thresholds,
)

SCRATCH = Path(__file__).parent


def load_labels(partition):
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    emb = np.load(SCRATCH / f"beats_{partition.replace('.csv', '')}.npz",
                  allow_pickle=True)
    x, keys = emb["embeddings"], [str(k) for k in emb["keys"]]
    row = {str(k): i for i, k in enumerate(npz["sample_key"])}
    idx = [row[k] for k in keys]
    y = (npz["Y_true"][idx] >= 0.5).astype(np.float32)
    m = npz["Y_mask"][idx].astype(np.float32)
    return x, y, m


class MLPHead(nn.Module):
    def __init__(self, d_in=768, d_hidden=512, p_drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.BatchNorm1d(d_hidden), nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden, 20),
        )

    def forward(self, x):
        return self.net(x)


def predict_probs(model, x):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), 1024):
            outs.append(torch.sigmoid(
                model(torch.from_numpy(x[s:s + 1024]).to(DEVICE))
            ).cpu().numpy())
    return np.concatenate(outs)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    x_all, y_all, m_all = load_labels("split01_train.csv")
    x_te, y_te, m_te = load_labels("split01_test.csv")
    tr_idx, val_idx = train_val_split(len(x_all))
    x_tr, y_tr, m_tr = x_all[tr_idx], y_all[tr_idx], m_all[tr_idx]
    x_val, y_val, m_val = x_all[val_idx], y_all[val_idx], m_all[val_idx]
    print(f"train {len(x_tr)} / val {len(x_val)} / test {len(x_te)}",
          flush=True)

    model = MLPHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    max_epochs, patience = 200, 20
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    x_tr_t = torch.from_numpy(x_tr)
    y_tr_t, m_tr_t = torch.from_numpy(y_tr), torch.from_numpy(m_tr)
    n, batch = len(x_tr), 256
    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            xb = x_tr_t[sel].to(DEVICE)
            yb, mb = y_tr_t[sel].to(DEVICE), m_tr_t[sel].to(DEVICE)
            opt.zero_grad()
            per = loss_fn(model(xb), yb)
            loss = (per * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()
        val_f1 = masked_f1(predict_probs(model, x_val), y_val, m_val,
                           0.5)["macro_f1"]
        if epoch % 10 == 0 or val_f1 > best_f1:
            print(f"  epoch {epoch:03d} loss {total / nb:.4f} "
                  f"val_f1 {val_f1:.4f}", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, since = val_f1, epoch, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= patience:
                break

    model.load_state_dict(best_state)
    minutes = (time.time() - t0) / 60
    test_probs = predict_probs(model, x_te)
    m_default = masked_f1(test_probs, y_te, m_te, 0.5)
    ths = tune_thresholds(predict_probs(model, x_val), y_val, m_val)
    m_tuned = masked_f1(test_probs, y_te, m_te, ths)

    torch.save(model.state_dict(), SCRATCH / "ckpt_E10_beats_head.pt")
    np.save(SCRATCH / "E10_test_probs.npy", test_probs)
    append_result({
        "name": "E10_beats_head", "minutes": round(minutes, 1),
        "best_val_f1": best_f1, "best_epoch": best_epoch,
        "test_macro_f1_at_0.5": m_default["macro_f1"],
        "test_macro_f1_tuned_th": m_tuned["macro_f1"],
        "tuned_thresholds": ths.tolist(),
        "per_class_f1_at_0.5": [c["f1"] for c in m_default["classes"]],
        "per_class_f1_tuned": [c["f1"] for c in m_tuned["classes"]],
    })
    print(f"E10_beats_head: test macro-F1 @0.5 = {m_default['macro_f1']:.4f},"
          f" tuned = {m_tuned['macro_f1']:.4f} ({minutes:.1f} min)",
          flush=True)
    print("BEATS HEAD DONE", flush=True)


if __name__ == "__main__":
    main()
