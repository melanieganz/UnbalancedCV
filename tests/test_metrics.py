# tests/test_metrics.py
from src import metrics

import numpy as np
import pytest


def test_accuracy_perfect_and_zero():
    y_true = np.array([0, 1, 1, 0])
    y_pred_perfect = np.array([0, 1, 1, 0])
    y_pred_none = np.array([1, 0, 0, 1])
    # perfect
    assert metrics.accuracy(y_true, y_pred_perfect, None) == 1.0
    # all wrong
    assert metrics.accuracy(y_true, y_pred_none, None) == 0.0
    # half right
    y_pred_half = np.array([0, 0, 1, 1])
    assert metrics.accuracy(y_true, y_pred_half, None) == 0.5


def test_auc_known_example():
    # from sklearn docs: metrics.auc should be 0.75
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    result = metrics.rocauc(y_true, None, y_score)
    assert pytest.approx(result, rel=1e-6) == 0.75


def test_aucprc_known_example():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    result = metrics.prcauc(y_true, None, y_scores)
    assert round(result, 2) == 0.83


def test_create_metrics():
    # Create a dictionary of metrics
    created_metrics = metrics.create_metrics(["accuracy", "rocauc", "prcauc"])

    # Check if the created metrics match the expected functions
    assert created_metrics["accuracy"] == metrics.accuracy
    assert created_metrics["rocauc"] == metrics.rocauc
    assert created_metrics["prcauc"] == metrics.prcauc


def test_create_metrics_invalid():
    # Test with an invalid metric name
    with pytest.raises(ValueError):
        _ = metrics.create_metrics(["invalid_metric"])
