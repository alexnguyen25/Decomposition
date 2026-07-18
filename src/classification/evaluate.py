"""Evaluate the trained OpenMIC classifier on the test partition."""

import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.classification.dataset import OpenMICDataset
from src.classification.model import Model
from src.config import (
    CACHE_DIR,
    CHECKPOINT_PATH,
    OPENMIC_DIR,
    TEST_PARTITION,
)


def setup() -> tuple[DataLoader, Model, torch.device]:
    """Create the test DataLoader and load the trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = OpenMICDataset(OPENMIC_DIR, TEST_PARTITION, CACHE_DIR)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = Model().to(device)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    return dataloader, model, device


def run_inference(
    dataloader: DataLoader,
    model: Model,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference and return test-set predictions, labels, and masks."""
    prediction_batches = []
    label_batches = []
    mask_batches = []

    with torch.no_grad():
        for spec, labels, label_mask in dataloader:
            logits = model(spec.to(device))
            predictions = (torch.sigmoid(logits) >= 0.5).cpu()

            prediction_batches.append(predictions)
            label_batches.append(labels)
            mask_batches.append(label_mask)

    predictions = torch.cat(prediction_batches, dim=0).numpy()
    labels = torch.cat(label_batches, dim=0).numpy()
    masks = torch.cat(mask_batches, dim=0).numpy()

    return predictions, labels, masks


def compute_metrics(predictions, labels, masks):
    """Per-class precision/recall/F1, only using confirmed labels. Also macro-F1."""
    results = []
    num_classes = predictions.shape[1]

    for c in range(num_classes):
        # only keep clips where this instrument's label was confirmed
        confirmed = masks[:, c] == 1
        y_pred = predictions[confirmed, c]
        y_true = labels[confirmed, c]

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        # avoid dividing by zero
        if tp + fp > 0:
            precision = tp / (tp + fp)
        else:
            precision = 0.0

        if tp + fn > 0:
            recall = tp / (tp + fn)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results.append({
            "class_index": c,
            "n_confirmed": int(confirmed.sum()),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    f1_scores = [r["f1"] for r in results]
    macro_f1 = float(np.mean(f1_scores))
    return {"classes": results, "macro_f1": macro_f1}


def print_report(metrics):
    """Print a table of per-class scores, then macro-F1."""
    # class-map.json is {"accordion": 0, "banjo": 1, ...}
    # flip it so we can look up name by index
    with open(OPENMIC_DIR / "class-map.json") as f:
        class_map = json.load(f)
    index_to_name = {index: name for name, index in class_map.items()}

    print(f"{'instrument':<20} {'confirmed':>9} {'precision':>9} {'recall':>9} {'f1':>9}")
    print("-" * 60)

    for r in metrics["classes"]:
        name = index_to_name[r["class_index"]]
        n = r["n_confirmed"]
        p = r["precision"]
        recall = r["recall"]
        f1 = r["f1"]
        print(f"{name:<20} {n:>9} {p:>9.3f} {recall:>9.3f} {f1:>9.3f}")

    print("-" * 60)
    print(f"Macro-F1: {metrics['macro_f1']:.3f}")


def main():
    """Run the complete evaluation workflow."""
    dataloader, model, device = setup()
    predictions, labels, masks = run_inference(dataloader, model, device)
    metrics = compute_metrics(predictions, labels, masks)
    print_report(metrics)


if __name__ == "__main__":
    main()
