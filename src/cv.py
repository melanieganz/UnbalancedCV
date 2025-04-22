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
    and also compute each metric on the pooled predictions across all folds.

    Returns
    -------
    {
      "folds": [ {fold metrics}, ... ],
      "pooled": { aggregated metrics on all samples }
    }
    """
    CV = StratifiedKFold if stratified else KFold
    cv = CV(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_results = []
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
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

        # per‑fold metrics
        res = {"fold": fold, "n_train": len(train_idx), "n_test": len(test_idx)}
        for name, fn in metrics.items():
            res[name] = fn(y_te, y_pred, y_proba)
        fold_results.append(res)

    # compute pooled metrics
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_proba_all = np.array(y_proba_all) if y_proba_all else None

    pooled = {"n_total": len(y_true_all)}
    for name, fn in metrics.items():
        pooled[name] = fn(y_true_all, y_pred_all, y_proba_all)

    return {"folds": fold_results, "pooled": pooled}
