"""E0: evaluate the existing checkpoint (models/classifier.pt) on cached test mels."""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from experiment import (  # noqa: E402
    DEVICE, OPENMIC_DIR, PROJECT, BaselineModel, load_partition, masked_f1, predict,
)


def main():
    x_te, y_te, m_te, _ = load_partition("split01_test.csv")
    print(f"test clips: {len(x_te)}", flush=True)

    model = BaselineModel().to(DEVICE)
    state = torch.load(PROJECT / "models" / "classifier.pt",
                       map_location=DEVICE, weights_only=True)
    # src model uses attribute names conv1/conv2/fc — same as BaselineModel
    model.load_state_dict(state)

    probs = predict(model, x_te, normalize=False)
    metrics = masked_f1(probs, y_te, m_te, 0.5)

    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    name_of = {i: n for n, i in class_map.items()}

    for c in metrics["classes"]:
        print(f"{name_of[c['class_index']]:<20} "
              f"P {c['precision']:.3f} R {c['recall']:.3f} F1 {c['f1']:.3f}")
    print(f"\nE0 baseline test macro-F1: {metrics['macro_f1']:.4f}")

    np.save(Path(__file__).parent / "baseline_per_class_f1.npy",
            np.array([c["f1"] for c in metrics["classes"]]))


if __name__ == "__main__":
    main()
