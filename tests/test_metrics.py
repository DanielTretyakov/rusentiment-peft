"""Unit tests for metrics."""
import numpy as np
from src.training.metrics import compute_metrics, full_report


def test_compute_metrics_perfect():
    labels = np.array([0, 1, 2, 0, 1])
    logits = np.array([
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
        [0.0, 0.0, 5.0],
        [5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0],
    ])
    result = compute_metrics((logits, labels))
    assert result["accuracy"] == 1.0
    assert result["f1_weighted"] == 1.0


def test_compute_metrics_shape():
    labels = np.array([0, 1, 2])
    logits = np.random.rand(3, 3)
    result = compute_metrics((logits, labels))
    assert "accuracy" in result
    assert "f1_weighted" in result
    assert "f1_macro" in result


def test_full_report_runs():
    labels = [0, 1, 2, 0, 1, 2]
    preds  = [0, 1, 1, 0, 2, 2]
    report = full_report(labels, preds)
    assert "negative" in report
    assert "positive" in report
