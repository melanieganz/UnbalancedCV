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
        "auc": auc,
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
    return float(np.mean(y_pred == y_true))


def auc(y_true: np.ndarray, y_pred: np.ndarray | None, y_proba: np.ndarray) -> float:
    """
    ROC‑AUC based on predicted scores.
    """
    fpr, tpr, _ = _sk_metrics.roc_curve(y_true, y_proba)
    return float(_sk_metrics.auc(fpr, tpr))
