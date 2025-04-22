# tests/test_cv.py

import os
import sys
import numpy as np
import pytest

# allow importing from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cv import run_cv
from metrics import accuracy, auc


class SimpleModel:
    """A “perfect” 0–1 classifier: learns mapping x→y exactly."""

    def fit(self, X, y):
        self.mapping = {tuple(x): int(label) for x, label in zip(X, y)}

    def predict(self, X):
        return np.array([self.mapping.get(tuple(x), 0) for x in X], dtype=int)

    def predict_proba(self, X):
        return np.array([[1 - self.mapping.get(tuple(x), 0), self.mapping.get(tuple(x), 0)] for x in X])


def test_run_cv_perfect_classifier():
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])

    metrics = {"accuracy": accuracy, "auc": auc}
    model = SimpleModel()

    result = run_cv(model, X, y, metrics, n_splits=2, stratified=True, random_state=0)
    folds = result["folds"]
    pooled = result["pooled"]

    # Should have 2 folds
    assert isinstance(folds, list)
    assert len(folds) == 2

    # Each fold is perfect
    for res in folds:
        assert set(res.keys()) >= {"fold", "n_train", "n_test", "accuracy", "auc"}
        assert res["accuracy"] == 1.0
        assert pytest.approx(res["auc"], rel=1e-9) == 1.0

    # Pooled across all folds is also perfect
    assert pooled["accuracy"] == 1.0
    assert pytest.approx(pooled["auc"], rel=1e-9) == 1.0


def test_run_cv_stratification_effect():
    # 50 samples: 45 negatives, 5 positives
    X = np.array([[0]] * 45 + [[1]] * 5)
    y = np.array([0] * 45 + [1] * 5)

    metrics = {"accuracy": accuracy, "auc": auc}
    model = SimpleModel()

    non_strat_result = run_cv(model, X, y, metrics, n_splits=5, stratified=False, random_state=0)
    strat_result = run_cv(model, X, y, metrics, n_splits=5, stratified=True, random_state=0)

    non_strat_folds = non_strat_result["folds"]
    strat_folds = strat_result["folds"]
    non_strat_pooled = non_strat_result["pooled"]
    strat_pooled = strat_result["pooled"]

    # Stratified: exactly 1 positive per fold → perfect AUC each fold
    for res in strat_folds:
        assert pytest.approx(res["auc"], rel=1e-9) == 1.0
    # Stratified pooled is also perfect
    assert pytest.approx(strat_pooled["auc"], rel=1e-9) == 1.0

    # Non‑stratified: at least one fold has zero positives → AUC nan
    assert any(np.isnan(res["auc"]) for res in non_strat_folds)
    # But pooled over all folds recovers perfect AUC
    assert non_strat_pooled["accuracy"] == 1.0
    assert pytest.approx(non_strat_pooled["auc"], rel=1e-9) == 1.0
