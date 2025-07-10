from src.cv import run_cv
from sklearn.linear_model import LogisticRegression

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def experiment_lung_cancer(metrics: list = ["accuracy", "rocauc"], seed: int = 123, verbose: bool = True) -> None:
    """
    Compare the effect of class imbalance on model performance.
    Use a simple logistic regression model and evaluate accuracy and AUC.
    """

    X, y = load_breast_cancer(return_X_y=True)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run cross-validation with original balanced dataset
    results = run_cv(
        LogisticRegression(max_iter=1000),
        X_scaled,
        y,
        metrics,
        n_splits=5,
        stratified=True,
        random_state=seed,
    )

    # print
    if verbose:
        # Print dataset characteristics
        print(f"Dataset shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Class balance: {np.bincount(y)[1] / len(y):.2f}")

        # Print results
        for result_type, result in results.items():
            print(f"{result_type.capitalize()} metrics:")
            for name, value in result.items():
                print(f"  {name}: {value:.4f}")

    return results
