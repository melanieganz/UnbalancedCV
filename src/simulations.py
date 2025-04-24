# src/simulations.py

import numpy as np


def simulate_dataset(
    n_samples: int,
    pos_ratio: float,
    mu0: float = -1,
    sigma0: float = 1,
    mu1: float = 1,
    sigma1: float = 1,
    seed: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a 1D two-class Gaussian mixture with given class imbalance.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    pos_ratio : float
        Proportion of positive class (label 1).
    mu0, sigma0 : float
        Mean and std dev for class 0 distribution.
    mu1, sigma1 : float
        Mean and std dev for class 1 distribution.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 1)
        Feature values.
    y : ndarray of shape (n_samples,)
        Binary labels {0, 1}.
    """
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, pos_ratio, size=n_samples)
    x = np.empty(n_samples)

    idx0 = y == 0
    idx1 = y == 1

    x[idx0] = rng.normal(mu0, sigma0, size=idx0.sum())
    x[idx1] = rng.normal(mu1, sigma1, size=idx1.sum())

    return x.reshape(-1, 1), y
