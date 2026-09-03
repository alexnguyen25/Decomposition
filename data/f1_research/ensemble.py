"""E8: ensemble the best mel-CNN, VGGish attention head, and PANNs head.

Averages test-set probabilities of available models, tunes per-class
thresholds for the ensemble on the validation split, evaluates on test.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRATCH = Path(__file__).parent
sys.path.insert(0, str(SCRATCH))
from experiment import (  # noqa: E402
    DEVICE, CNN4, append_result, load_partition, masked_f1, predict,
    train_val_split, tune_thresholds,
)
from vggish_attention import AttentionHead, load_vggish_seq  # noqa: E402


def best_cnn_record():
    best = None
    for line in open(SCRATCH / "results.jsonl"):
        r = json.loads(line)
        if r["name"].startswith(("E2", "E3", "E4")):
            if best is None or (r["test_macro_f1_tuned_th"]
                                > best["test_macro_f1_tuned_th"]):
                best = r
    return best


def main():
    val_probs_list, test_probs_list, members = [], [], []

    # --- mel CNN (best of E2-E4)
    best = best_cnn_record()
    x_all, y_all, m_all, _ = load_partition("split01_train.csv")
    tr_idx, val_idx = train_val_split(len(x_all))
    x_val, y_val, m_val = x_all[val_idx], y_all[val_idx], m_all[val_idx]
    x_te, y_te, m_te, _ = load_partition("split01_test.csv")

    cnn = CNN4().to(DEVICE)
    cnn.load_state_dict(torch.load(SCRATCH / f"ckpt_{best['name']}.pt",
                                   map_location=DEVICE, weights_only=True))
    cnn.eval()
    val_probs_list.append(predict(cnn, x_val, normalize=True))
    test_probs_list.append(predict(cnn, x_te, normalize=True))
    members.append(best["name"])
    del x_all, x_te

    # --- VGGish attention head (E6)
    xv_all, yv_all, mv_all = load_vggish_seq("split01_train.csv")
    xv_te, _, _ = load_vggish_seq("split01_test.csv")
    att = AttentionHead().to(DEVICE)
    att.load_state_dict(torch.load(SCRATCH / "ckpt_E6_vggish_attention.pt",
                                   map_location=DEVICE, weights_only=True))
    att.eval()
    with torch.no_grad():
        pv_val = torch.sigmoid(att(torch.from_numpy(xv_all[val_idx]).to(DEVICE))
                               ).cpu().numpy()
        pv_te = torch.sigmoid(att(torch.from_numpy(xv_te).to(DEVICE))
                              ).cpu().numpy()
    val_probs_list.append(pv_val)
    test_probs_list.append(pv_te)
    members.append("E6_vggish_attention")

    # --- PANNs head (E7), if available
    e7_ckpt = SCRATCH / "ckpt_E7_panns_head.pt"
    if e7_ckpt.exists():
        from panns_head import MLPHead, load_labels, predict_probs
        xp_all, _, _ = load_labels("split01_train.csv")
        xp_te, _, _ = load_labels("split01_test.csv")
        head = MLPHead().to(DEVICE)
        head.load_state_dict(torch.load(e7_ckpt, map_location=DEVICE,
                                        weights_only=True))
        head.eval()
        val_probs_list.append(predict_probs(head, xp_all[val_idx]))
        test_probs_list.append(predict_probs(head, xp_te))
        members.append("E7_panns_head")

    val_probs = np.mean(val_probs_list, axis=0)
    test_probs = np.mean(test_probs_list, axis=0)

    m_default = masked_f1(test_probs, y_te, m_te, 0.5)
    ths = tune_thresholds(val_probs, y_val, m_val)
    m_tuned = masked_f1(test_probs, y_te, m_te, ths)

    append_result({
        "name": "E8_ensemble", "members": members,
        "test_macro_f1_at_0.5": m_default["macro_f1"],
        "test_macro_f1_tuned_th": m_tuned["macro_f1"],
        "tuned_thresholds": ths.tolist(),
        "per_class_f1_at_0.5": [c["f1"] for c in m_default["classes"]],
        "per_class_f1_tuned": [c["f1"] for c in m_tuned["classes"]],
    })
    print(f"E8_ensemble ({' + '.join(members)}): "
          f"test macro-F1 @0.5 = {m_default['macro_f1']:.4f}, "
          f"tuned = {m_tuned['macro_f1']:.4f}", flush=True)
    print("ENSEMBLE DONE", flush=True)


if __name__ == "__main__":
    main()
