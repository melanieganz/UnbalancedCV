# tests/test_cv.py
from src.cv import run_cv
from src.metrics import accuracy, rocauc

import numpy as np
import pytest


class SimpleModel:
    """A “perfect” 0–1 classifier: learns mapping x→y exactly."""

    def fit(self, X, y):
        self.mapping = {tuple(x): int(label) for x, label in zip(X, y)}

    def predict(self, X):
        return np.array([self.mapping.get(tuple(x), 0) for x in X], dtype=int)

    def predict_proba(self, X):
        return np.array([[1 - self.mapping.get(tuple(x), 0), self.mapping.get(tuple(x), 0)] for x in X])


def test_run_cv_perfect_classifier():
    # Simple dataset where feature == label
    X = np.array([[0], [1], [0], [1]])
    y = np.array([0, 1, 0, 1])

    metrics = {"accuracy": accuracy, "rocauc": rocauc}
    model = SimpleModel()

    result = run_cv(
        model=model,
        X=X,
        y=y,
        metrics=metrics,
        n_splits=2,
        stratified=True,
        random_state=0,
    )

    avg = result["average"]
    pooled = result["pooled"]

    # Fold‐average should be perfect
    assert avg["accuracy"] == pytest.approx(1.0)
    assert avg["rocauc"] == pytest.approx(1.0)

    # Pooled over all folds also perfect
    assert pooled["accuracy"] == pytest.approx(1.0)
    assert pooled["rocauc"] == pytest.approx(1.0)


def test_run_cv_stratification_effect():
    # 50 samples: 45 negatives (0), 5 positives (1)
    X = np.array([[0]] * 45 + [[1]] * 5)
    y = np.array([0] * 45 + [1] * 5)

    metrics = {"accuracy": accuracy, "rocauc": rocauc}
    model = SimpleModel()

    non_strat_res = run_cv(
        model=model,
        X=X,
        y=y,
        metrics=metrics,
        n_splits=5,
        stratified=False,
        random_state=0,
    )
    strat_res = run_cv(
        model=model,
        X=X,
        y=y,
        metrics=metrics,
        n_splits=5,
        stratified=True,
        random_state=0,
    )

    avg_non = non_strat_res["average"]
    pooled_non = non_strat_res["pooled"]

    avg_strat = strat_res["average"]
    pooled_strat = strat_res["pooled"]

    # Stratified: exactly one positive per fold → perfect AUC everywhere
    assert avg_strat["accuracy"] == pytest.approx(1.0)
    assert avg_strat["rocauc"] == pytest.approx(1.0)
    assert pooled_strat["accuracy"] == pytest.approx(1.0)
    assert pooled_strat["rocauc"] == pytest.approx(1.0)

    # Non‑stratified: some folds lack positives → fold‐average AUC is nan
    assert avg_non["accuracy"] == pytest.approx(1.0)
    assert np.isnan(avg_non["rocauc"])

    # But pooled over all folds recovers perfect AUC
    assert pooled_non["accuracy"] == pytest.approx(1.0)
    assert pooled_non["rocauc"] == pytest.approx(1.0)
