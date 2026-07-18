"""Tests for mask-aware classifier evaluation metrics."""

import unittest

import numpy as np

from src.classification.evaluate import compute_metrics


class TestComputeMetrics(unittest.TestCase):
    def test_computes_per_class_metrics_using_only_confirmed_labels(self):
        predictions = np.array(
            [[1, 0], [1, 0], [0, 0], [0, 0]], dtype=np.int64
        )
        labels = np.array(
            [[1, 1], [0, 0], [1, 1], [0, 0]], dtype=np.int64
        )
        masks = np.array(
            [[1, 1], [1, 1], [0, 1], [1, 0]], dtype=np.int64
        )

        metrics = compute_metrics(predictions, labels, masks)

        first_class = metrics["classes"][0]
        self.assertEqual(first_class["class_index"], 0)
        self.assertEqual(first_class["n_confirmed"], 3)
        self.assertAlmostEqual(first_class["precision"], 0.5)
        self.assertAlmostEqual(first_class["recall"], 1.0)
        self.assertAlmostEqual(first_class["f1"], 2 / 3)

    def test_returns_zero_when_metric_denominators_are_zero(self):
        predictions = np.zeros((3, 1), dtype=np.int64)
        labels = np.array([[1], [0], [1]], dtype=np.int64)
        masks = np.ones((3, 1), dtype=np.int64)

        metrics = compute_metrics(predictions, labels, masks)

        only_class = metrics["classes"][0]
        self.assertEqual(only_class["precision"], 0.0)
        self.assertEqual(only_class["recall"], 0.0)
        self.assertEqual(only_class["f1"], 0.0)
        self.assertEqual(metrics["macro_f1"], 0.0)


if __name__ == "__main__":
    unittest.main()
