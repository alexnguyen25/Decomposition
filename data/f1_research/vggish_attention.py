"""E6: per-class attention pooling over VGGish timesteps (Gururani-style).

Instead of mean+max pooling the (10, 128) sequence, learn per-class attention
weights over timesteps so each instrument attends to the segments where it is
audible. Multi-instance learning view of the weak clip-level labels.
"""

import json
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
VGGISH_DIR = OPENMIC_DIR / "vggish"


def load_vggish_seq(partition: str):
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    y_true, y_mask, sample_key = npz["Y_true"], npz["Y_mask"], npz["sample_key"]
    with open(OPENMIC_DIR / "partitions" / partition) as f:
        keys = [line.strip() for line in f if line.strip()]
    keyset = set(keys)
    sel = np.array([k in keyset for k in sample_key])

    y = (y_true[sel] >= 0.5).astype(np.float32)
    m = y_mask[sel].astype(np.float32)
    kept = sample_key[sel]

    x = np.zeros((sel.sum(), 10, 128), dtype=np.float32)
    for i, key in enumerate(kept):
        with open(VGGISH_DIR / key[:3] / f"{key}.json") as f:
            feats = np.array(json.load(f)["features"], dtype=np.float32) / 255.0
        t = min(feats.shape[0], 10)
        x[i, :t] = feats[:t]
    return x, y, m


class AttentionHead(nn.Module):
    def __init__(self, d_in=128, d_hidden=256, n_classes=20, p_drop=0.4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Dropout(p_drop),
        )
        self.att = nn.Linear(d_hidden, n_classes)   # attention scores per class
        self.cls = nn.Linear(d_hidden, n_classes)   # per-timestep class logits

    def forward(self, x):                # x: (B, 10, 128)
        h = self.embed(x)                # (B, 10, H)
        a = torch.softmax(self.att(h), dim=1)   # (B, 10, C) over timesteps
        z = self.cls(h)                  # (B, 10, C)
        return (a * z).sum(dim=1)        # (B, C)


def predict_probs(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).to(DEVICE))).cpu().numpy()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("loading vggish sequences...", flush=True)
    x_all, y_all, m_all = load_vggish_seq("split01_train.csv")
    x_te, y_te, m_te = load_vggish_seq("split01_test.csv")
    tr_idx, val_idx = train_val_split(len(x_all))
    x_tr, y_tr, m_tr = x_all[tr_idx], y_all[tr_idx], m_all[tr_idx]
    x_val, y_val, m_val = x_all[val_idx], y_all[val_idx], m_all[val_idx]

    model = AttentionHead().to(DEVICE)
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
            selb = perm[s:s + batch]
            xb = x_tr_t[selb].to(DEVICE)
            yb, mb = y_tr_t[selb].to(DEVICE), m_tr_t[selb].to(DEVICE)
            opt.zero_grad()
            per_el = loss_fn(model(xb), yb)
            loss = (per_el * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        sched.step()

        val_f1 = masked_f1(predict_probs(model, x_val), y_val, m_val, 0.5)["macro_f1"]
        if epoch % 10 == 0 or val_f1 > best_f1:
            print(f"  epoch {epoch:03d} loss {total / nb:.4f} val_f1 {val_f1:.4f}",
                  flush=True)
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

    torch.save(model.state_dict(), SCRATCH / "ckpt_E6_vggish_attention.pt")
    append_result({
        "name": "E6_vggish_attention", "minutes": round(minutes, 1),
        "best_val_f1": best_f1, "best_epoch": best_epoch,
        "test_macro_f1_at_0.5": m_default["macro_f1"],
        "test_macro_f1_tuned_th": m_tuned["macro_f1"],
        "tuned_thresholds": ths.tolist(),
        "per_class_f1_at_0.5": [c["f1"] for c in m_default["classes"]],
        "per_class_f1_tuned": [c["f1"] for c in m_tuned["classes"]],
    })
    print(f"E6_vggish_attention: test macro-F1 @0.5 = {m_default['macro_f1']:.4f}, "
          f"tuned = {m_tuned['macro_f1']:.4f} ({minutes:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
