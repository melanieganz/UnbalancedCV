# tests/test_simulations.py

import os
import sys
import numpy as np
import pytest

# so we can import simulations.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from simulations import simulate_dataset


def test_shapes_and_types():
    X, y = simulate_dataset(n_samples=50, pos_ratio=0.3, seed=123)
    # X should be (50, 1), y should be (50,)
    assert X.shape == (50, 1)
    assert y.shape == (50,)
    # y values must be 0 or 1 and integer type
    assert set(np.unique(y)).issubset({0, 1})
    assert np.issubdtype(y.dtype, np.integer)


def test_reproducibility_with_seed():
    X1, y1 = simulate_dataset(100, 0.2, seed=0)
    X2, y2 = simulate_dataset(100, 0.2, seed=0)
    # exact match when using same seed
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_randomness_without_seed():
    X1, y1 = simulate_dataset(100, 0.2, seed=None)
    X2, y2 = simulate_dataset(100, 0.2, seed=None)
    # extremely unlikely to match exactly by chance
    with pytest.raises(AssertionError):
        assert np.array_equal(y1, y2)
