from src.simulations import simulate_dataset
from src.cv import run_cv
from src.metrics import create_metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


def experiment_simulation(
    pos_ratio: float = 0.1,
    n_samples: int = 1000,
    mu0: float = -1,
    sigma0: float = 1,
    mu1: float = 1,
    sigma1: float = 1,
    eval_metrics: list = ["accuracy", "auc"],
    seed: int = 123,
    verbose: bool = True,
) -> None:
    """
    Compare the effect of class imbalance on model performance.
    Use a simple logistic regression model and evaluate accuracy and AUC.
    """

    # Simulate dataset
    X, y = simulate_dataset(n_samples, pos_ratio, mu0, sigma0, mu1, sigma1, seed=seed)

    if verbose:
        # plot the dataset
        fig, ax = plt.subplots()
        ax.set_title(f"Simulated dataset (pos_ratio={pos_ratio}, n_samples={n_samples})")
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Count")
        ax.hist(X[y == 0], bins=30, alpha=0.5, label="Class 0", color="blue")
        ax.hist(X[y == 1], bins=30, alpha=0.5, label="Class 1", color="orange")
        ax.legend()
        plt.show()

    # Define model and metrics
    model = LogisticRegression()

    metrics = create_metrics(eval_metrics)

    # Run cross-validation
    results = run_cv(model, X, y, metrics, n_splits=5, stratified=True, random_state=seed)

    # Simulate a new dataset for generalization
    X_gen, y_gen = simulate_dataset(1000, 0.1, mu0, sigma0, mu1, sigma1, seed=seed * 100)

    # fit the model on entire training set
    clf_full = LogisticRegression(solver="lbfgs").fit(X, y)

    # Predict and score on fresh ("generalization") data
    y_pred_gen = clf_full.predict(X_gen)
    y_proba_gen = clf_full.predict_proba(X_gen)[:, 1]

    # add generalization to the results
    results["generalized"] = {}
    for name, function in metrics.items():
        score = function(y_gen, y_pred_gen, y_proba_gen)
        results["generalized"][name] = score

    if verbose:
        for result_type, result in results.items():
            print(f"{result_type.capitalize()} metrics:")
            for name, value in result.items():
                print(f"  {name}: {value:.4f}")

    return results


def experiment_lung_cancer(metrics: list = ["accuracy", "auc"], seed: int = 123, verbose: bool = True) -> None:
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
        LogisticRegression(max_iter=1000), X_scaled, y, metrics, n_splits=5, stratified=True, random_state=seed
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
