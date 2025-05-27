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
) -> dict:
    """
    Perform K‑fold (optionally stratified) CV, compute each metric per fold,
    then return:
      - average: dict of average metric over folds
      - pooled:  dict of metric computed on all test‑fold predictions concatenated
    """
    CV = StratifiedKFold if stratified else KFold
    cv = CV(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_te)[:, 1]

        # accumulate for pooling
        y_true_all.extend(y_te)
        y_pred_all.extend(y_pred)
        if y_proba is not None:
            y_proba_all.extend(y_proba)

        # per‑fold metric values only (we'll average below)
        fold_results.append({name: fn(y_te, y_pred, y_proba) for name, fn in metrics.items()})

    # compute average across folds
    average = {name: float(np.mean([fr[name] for fr in fold_results])) for name in metrics.keys()}

    # compute pooled metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all) if y_proba_all else None

    pooled = {}
    for name, fn in metrics.items():
        pooled[name] = fn(y_true_all, y_pred_all, y_proba_all)

    return {"average": average, "pooled": pooled}
