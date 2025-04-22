# src/cv.py

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
from typing import Callable, Mapping

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray | None], float]


def run_cv(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metrics: Mapping[str, MetricFn],
    n_splits: int = 5,
    stratified: bool = False,
    random_state: int = 0,
) -> list[dict]:
    """
    Perform K‑fold (optionally stratified) CV and compute arbitrary metrics.

    Parameters
    ----------
    model
        Any sklearn‐style estimator with .fit, .predict, .predict_proba
    X, y
        Data.
    metrics
        Dict mapping metric‐name → function(y_true, y_pred, y_proba).
    n_splits
        Number of folds.
    stratified
        If True, use StratifiedKFold; else plain KFold.
    random_state
        For shuffle reproducibility.

    Returns
    -------
    results : list of dict
        One dict per fold, with keys:
        - 'fold', 'n_train', 'n_test'
        - one key per metric name, with its score.
    """
    CV = StratifiedKFold if stratified else KFold
    cv = CV(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        # some metrics (like AUC) need probabilities:
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)[:, 1]

        res = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        for name, fn in metrics.items():
            res[name] = fn(y_te, y_pred, y_proba)
        all_results.append(res)

    return all_results
