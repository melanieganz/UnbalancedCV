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
    Generate a 1D two‐class Gaussian mixture with given class imbalance.

    Returns
    -------
    X : array, shape (n_samples,1)
        Feature values.
    y : array, shape (n_samples,)
        Binary labels {0,1}.
    """
    if seed is not None:
        np.random.seed(seed)
    y = np.random.binomial(1, pos_ratio, size=n_samples)
    x = np.where(
        y == 0,
        np.random.normal(mu0, sigma0, size=n_samples),
        np.random.normal(mu1, sigma1, size=n_samples),
    )
    return x.reshape(-1, 1), y
