# src/metrics.py

import numpy as np
from sklearn import metrics as _sk_metrics


def create_metrics(metric_names: list[str]) -> dict[str, callable]:
    """
    Create a dictionary of metrics based on the provided list of metric names.

    :param metric_names: List of metric names to include in the dictionary.
    :return: Dictionary of metric functions.
    """
    available_metrics = {
        "accuracy": accuracy,
        "rocauc": rocauc,
    }

    metrics = {}
    for name in metric_names:
        if name in available_metrics:
            metrics[name] = available_metrics[name]
        else:
            raise ValueError(f"Metric '{name}' is not available.")
    return metrics


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> float:
    """
    Fraction of exact matches.
    """
    return _sk_metrics.accuracy_score(y_true, y_pred)


def rocauc(y_true: np.ndarray, y_pred: np.ndarray | None, y_proba: np.ndarray) -> float:
    """
    ROC‑AUC based on predicted scores.
    """
    return _sk_metrics.roc_auc_score(y_true, y_proba)
