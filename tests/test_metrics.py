# tests/test_metrics.py
from src.metrics import accuracy, auc

import numpy as np
import pytest


def test_accuracy_perfect_and_zero():
    y_true = np.array([0, 1, 1, 0])
    y_pred_perfect = np.array([0, 1, 1, 0])
    y_pred_none = np.array([1, 0, 0, 1])
    # perfect
    assert accuracy(y_true, y_pred_perfect, None) == 1.0
    # all wrong
    assert accuracy(y_true, y_pred_none, None) == 0.0
    # half right
    y_pred_half = np.array([0, 0, 1, 1])
    assert accuracy(y_true, y_pred_half, None) == 0.5


def test_auc_known_example():
    # from sklearn docs: AUC should be 0.75
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    result = auc(y_true, None, y_score)
    assert pytest.approx(result, rel=1e-6) == 0.75


def test_auc_errors_on_missing_proba():
    y_true = np.array([0, 1])
    # if y_proba is None, roc_curve will error
    with pytest.raises(ValueError):
        _ = auc(y_true, None, None)
