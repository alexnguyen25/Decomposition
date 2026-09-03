"""E9: fine-tune CNN14 end-to-end on OpenMIC (from cached logmels).

Phase 1: linear head only (backbone frozen), 2 epochs.
Phase 2: unfreeze conv_block4-6 + fc1 (early blocks kept in eval mode so
their BN stats stay frozen), low LR, early stop on val macro-F1.

Same val split / metric as every other experiment. Appends to results.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
sys.path.insert(0, str(SCRATCH / "pylibs"))
from experiment import (  # noqa: E402
    DEVICE, OPENMIC_DIR, SEED, append_result, masked_f1, train_val_split,
    tune_thresholds,
)
from panns_inference.models import Cnn14  # noqa: E402

LOGMELS = SCRATCH / "cnn14_logmel_cache"
T_FRAMES = 1001


class FineTuneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Cnn14(sample_rate=32000, window_size=1024,
                              hop_size=320, mel_bins=64, fmin=50, fmax=14000,
                              classes_num=527)
        self.backbone.load_state_dict(
            torch.load(SCRATCH / "Cnn14_mAP=0.431.pth",
                       map_location="cpu")["model"])
        self.head = nn.Linear(2048, 20)

    def forward(self, x):                     # x: (B, T, 64) logmel
        m = self.backbone
        x = x.unsqueeze(1)                    # (B,1,T,64)
        x = x.transpose(1, 3)
        x = m.bn0(x)
        x = x.transpose(1, 3)
        if self.training:
            x = m.spec_augmenter(x)
        for i, blk in enumerate([m.conv_block1, m.conv_block2, m.conv_block3,
                                 m.conv_block4, m.conv_block5]):
            x = blk(x, pool_size=(2, 2), pool_type="avg")
            x = F.dropout(x, p=0.2, training=self.training)
        x = m.conv_block6(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = torch.mean(x, dim=3)
        (x1, _) = torch.max(x, dim=2)
        x = x1 + torch.mean(x, dim=2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(m.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.head(x)

    def set_trainable(self, phase):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = True
        if phase == 2:
            for mod in (self.backbone.conv_block4, self.backbone.conv_block5,
                        self.backbone.conv_block6, self.backbone.fc1):
                for p in mod.parameters():
                    p.requires_grad = True

    def frozen_blocks_eval(self, phase):
        """Keep frozen blocks' BN stats fixed while training."""
        frozen = [self.backbone.bn0, self.backbone.conv_block1,
                  self.backbone.conv_block2, self.backbone.conv_block3]
        if phase == 1:
            frozen += [self.backbone.conv_block4, self.backbone.conv_block5,
                       self.backbone.conv_block6]
        for mod in frozen:
            mod.eval()


def load_split(partition):
    npz = np.load(OPENMIC_DIR / "openmic-2018.npz", allow_pickle=True)
    row = {str(k): i for i, k in enumerate(npz["sample_key"])}
    with open(OPENMIC_DIR / "partitions" / partition) as f:
        keys = [l.strip() for l in f if l.strip()]
    idx = [row[k] for k in keys]
    y = (npz["Y_true"][idx] >= 0.5).astype(np.float32)
    m = npz["Y_mask"][idx].astype(np.float32)
    x = np.zeros((len(keys), T_FRAMES, 64), dtype=np.float16)
    for i, k in enumerate(keys):
        a = np.load(LOGMELS / f"{k}.npy")
        t = min(a.shape[0], T_FRAMES)
        x[i, :t] = a[:t]
    return x, y, m


def predict(model, x, batch=64):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch):
            xb = torch.from_numpy(x[s:s + batch].astype(np.float32)).to(DEVICE)
            outs.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(outs)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    print("loading logmels...", flush=True)
    x_all, y_all, m_all = load_split("split01_train.csv")
    tr_idx, val_idx = train_val_split(len(x_all))
    x_tr, y_tr, m_tr = x_all[tr_idx], y_all[tr_idx], m_all[tr_idx]
    x_val, y_val, m_val = x_all[val_idx], y_all[val_idx], m_all[val_idx]
    print(f"train {len(x_tr)} / val {len(x_val)}", flush=True)

    model = FineTuneModel().to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    y_tr_t, m_tr_t = torch.from_numpy(y_tr), torch.from_numpy(m_tr)
    n, batch = len(x_tr), 32
    best_f1, best_state, best_epoch, since = -1.0, None, -1, 0
    t0 = time.time()
    history = []

    def run_epoch(opt, phase, epoch):
        nonlocal best_f1, best_state, best_epoch, since
        model.train()
        model.frozen_blocks_eval(phase)
        perm = np.random.permutation(n)
        tot = nb = 0
        for s in range(0, n, batch):
            sel = perm[s:s + batch]
            xb = torch.from_numpy(x_tr[sel].astype(np.float32)).to(DEVICE)
            yb, mb = y_tr_t[sel].to(DEVICE), m_tr_t[sel].to(DEVICE)
            opt.zero_grad()
            per = loss_fn(model(xb), yb)
            loss = (per * mb).sum() / mb.sum()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        val_f1 = masked_f1(predict(model, x_val), y_val, m_val,
                           0.5)["macro_f1"]
        history.append({"phase": phase, "epoch": epoch,
                        "loss": tot / nb, "val_f1": val_f1})
        print(f"  P{phase} epoch {epoch} loss {tot / nb:.4f} "
              f"val_f1 {val_f1:.4f} ({(time.time() - t0) / 60:.0f}m)",
              flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, since = val_f1, len(history), 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            since += 1

    model.set_trainable(1)
    opt1 = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3,
        weight_decay=1e-4)
    for e in range(2):
        run_epoch(opt1, 1, e)

    model.set_trainable(2)
    opt2 = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": 3e-4},
        {"params": [p for mod in (model.backbone.conv_block4,
                                  model.backbone.conv_block5,
                                  model.backbone.conv_block6,
                                  model.backbone.fc1)
                    for p in mod.parameters()], "lr": 1e-5},
    ], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)
    for e in range(10):
        run_epoch(opt2, 2, e)
        sched.step()
        if since >= 3:
            print("  early stop", flush=True)
            break

    model.load_state_dict(best_state)
    minutes = (time.time() - t0) / 60

    print("loading test logmels...", flush=True)
    x_te, y_te, m_te = load_split("split01_test.csv")
    test_probs = predict(model, x_te)
    m_default = masked_f1(test_probs, y_te, m_te, 0.5)
    ths = tune_thresholds(predict(model, x_val), y_val, m_val)
    m_tuned = masked_f1(test_probs, y_te, m_te, ths)

    torch.save(model.state_dict(), SCRATCH / "ckpt_E9_cnn14_finetune.pt")
    np.save(SCRATCH / "E9_test_probs.npy", test_probs)
    append_result({
        "name": "E9_cnn14_finetune", "minutes": round(minutes, 1),
        "best_val_f1": best_f1, "best_epoch": best_epoch,
        "test_macro_f1_at_0.5": m_default["macro_f1"],
        "test_macro_f1_tuned_th": m_tuned["macro_f1"],
        "tuned_thresholds": ths.tolist(),
        "per_class_f1_at_0.5": [c["f1"] for c in m_default["classes"]],
        "per_class_f1_tuned": [c["f1"] for c in m_tuned["classes"]],
        "history": history,
    })
    print(f"E9_cnn14_finetune: test macro-F1 @0.5 = "
          f"{m_default['macro_f1']:.4f}, tuned = {m_tuned['macro_f1']:.4f} "
          f"({minutes:.0f} min)", flush=True)
    print("FINETUNE DONE", flush=True)


if __name__ == "__main__":
    main()
