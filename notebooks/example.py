"""
Examples for UnbalancedCV
==========================
This script provides examples with real-world data to demonstrate the difference
between different cross-validation approaches.

Examples included:
1. Breast Cancer Prediction
2. Cognitive Impairment Classification
3. Depression Remission Prediction
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import our custom functions
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import create_metrics
from src.cv import run_cv


# Functions for plotting
def plot_fold_probs(y_true_folds, y_prob_folds, ax, label):
    """Plot predicted probabilities per fold."""
    for fold_idx, (y_true, y_prob) in enumerate(zip(y_true_folds, y_prob_folds)):
        y_true = y_true[0]
        y_prob = y_prob[0]
        # true
        ax.scatter(
            [fold_idx] * len(y_prob[y_true == 1]),
            y_prob[y_true == 1],
            c="blue",
            alpha=0.4,
            label=f"Fold {fold_idx + 1}" if fold_idx == 0 else "",
        )
        # false
        ax.scatter(
            [fold_idx] * len(y_prob[y_true == 0]),
            y_prob[y_true == 0],
            c="red",
            alpha=0.4,
            label=f"Fold {fold_idx + 1}" if fold_idx == 0 else "",
        )
    ax.axhline(0.5, color="gray", linestyle="--", label="Decision boundary")
    ax.set_xlabel(f"Fold\n{label}")
    ax.set_ylabel("Predicted probability")


def setup_plotting():
    """Configure matplotlib plotting style."""
    plt.rcParams.update(
        {
            "figure.dpi": 300,
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.2,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "text.usetex": True,  # Uncomment for LaTeX
            "axes.spines.top": True,
            "axes.spines.right": True,
        }
    )
    cm = 1 / 2.54
    fig_width = 17 * cm
    fig_height = 17 * cm / 3
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    return fig, axes


def example_breast_cancer(ax):
    """
    Example 1: Breast Cancer Prediction
    ====================================
    Classification task: predict whether the subject has cancer or not.
    """
    print("\n" + "=" * 60)
    print("Example 1: Breast Cancer Prediction")
    print("=" * 60)

    X, y = load_breast_cancer(return_X_y=True)

    # print dataset shape
    print(f"Dataset shape: {X.shape}")
    # Check class balance
    print(f"Class distribution:\n{sum(y)} positive, {len(y) - sum(y)} negative, ratio: {sum(y) / len(y):.2f}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run cross-validation
    model = LogisticRegression()
    metrics = create_metrics(["accuracy", "rocauc", "prcauc"])
    results = run_cv(model, X_scaled, y, metrics, n_splits=5, stratified=True, random_state=1)

    results_df = pd.DataFrame({"average": results["average"], "pooled": results["pooled"]})
    print(results_df)
    print(
        f"Difference between average and pooled AUC: {results_df['average']['rocauc'] - results_df['pooled']['rocauc']}"
    )

    # Plot results
    y_true_folds = results["true"]
    y_prob_folds = results["probs"]
    plot_fold_probs(y_true_folds, y_prob_folds, ax, "a)")


def example_cognitive_impairment(ax):
    """
    Example 2: Cognitive Impairment
    ================================
    Classification task: predict whether the subject is cognitively normal or not,
    based on cortical features from Oasis3 data.
    """
    print("\n" + "=" * 60)
    print("Example 2: Cognitive Impairment Classification")
    print("=" * 60)

    # Load the dataset
    data_path = Path(__file__).parent.parent / "data" / "oasis3_fs_mci.tsv"
    df = pd.read_csv(data_path, sep="\t")

    print(f"Dataset shape: {df.shape}")

    # drop rows that have empty cells / NAs
    df = df.dropna(axis=0, how="any")

    # only keep first occurence of each subject
    df_baseline = df.drop_duplicates(subset=["subject"], keep="first")
    print(f"Shape of baseline data: {df_baseline.shape}")

    # split into X
    X = df_baseline.drop(columns=["subject", "session", "age", "cognitiveyly_normal"])
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.to_numpy()

    # standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # and y
    y = df_baseline["cognitiveyly_normal"].to_numpy()
    y_binary = (y == True).astype(int)

    # Check class balance
    print(
        f"Class distribution:\n{sum(y_binary)} positive, {len(y_binary) - sum(y_binary)} negative, ratio: {sum(y_binary) / len(y_binary):.2f}"
    )

    # Run cross-validation
    model = LogisticRegression(max_iter=1000000)
    metrics = create_metrics(["accuracy", "rocauc", "prcauc"])
    results = run_cv(model, X_scaled, y_binary, metrics, n_splits=5, stratified=True, random_state=1)
    results_df = pd.DataFrame({"average": results["average"], "pooled": results["pooled"]})
    print(results_df)
    print(
        f"Difference between average and pooled AUC: {results_df['average']['rocauc'] - results_df['pooled']['rocauc']}"
    )

    # Plot probabilities
    plot_fold_probs(results["true"], results["probs"], ax, "b)")


def example_depression_remission(ax):
    """
    Example 3: Depression Remission
    ================================
    Classification task: predict whether a patient will achieve remission from
    depression after a certain period of time.
    """
    print("\n" + "=" * 60)
    print("Example 3: Depression Remission Prediction")
    print("=" * 60)

    # Load the dataset
    data_path = Path(__file__).parent.parent / "data" / "np1_fs_mdd_episode.csv"
    df = pd.read_csv(data_path, sep=",")

    print(f"Dataset shape: {df.shape}")

    # drop rows that have empty cells / NAs
    df = df.dropna(axis=0, how="any")

    print(f"Shape of baseline data: {df.shape}")

    X = df.drop(columns=["mdd_episode", "diagnosis"])
    X = X.to_numpy()

    # standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # and y
    y = df["mdd_episode"].to_numpy()
    y_binary = (y == "Recurrent").astype(int)

    # Check class balance
    print(
        f"Class distribution:\n{sum(y_binary)} positive, {len(y_binary) - sum(y_binary)} negative, ratio: {sum(y_binary) / len(y_binary):.2f}"
    )

    # Run cross-validation
    model = LogisticRegression(max_iter=1000000)
    metrics = create_metrics(["accuracy", "rocauc", "prcauc"])
    results = run_cv(model, X_scaled, y_binary, metrics, n_splits=5, stratified=True, random_state=1)
    results_df = pd.DataFrame({"average": results["average"], "pooled": results["pooled"]})
    print(results_df)
    print(
        f"Difference between average and pooled AUC: {results_df['average']['rocauc'] - results_df['pooled']['rocauc']}"
    )

    # plot results
    plot_fold_probs(results["true"], results["probs"], ax, "c)")


def main():
    """Run all examples and generate plots."""
    # Setup plotting
    fig, axes = setup_plotting()

    # Run all examples
    example_breast_cancer(axes[0])
    example_cognitive_impairment(axes[1])
    example_depression_remission(axes[2])

    # Finalize and show plot
    fig.tight_layout()
    plt.show()
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
