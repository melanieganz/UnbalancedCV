# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def compute_auc(data):
    """
    Compute AUC on pooled data and as weighted average across folds.

    Parameters
    ----------
    data : dict
        Output from simulate_cv_data()

    Returns
    -------
    dict
        Dictionary containing:
        - 'auc_pooled': AUC computed on all data pooled together
        - 'auc_weighted': Weighted average of per-fold AUCs (weighted by Nk/N)
        - 'auc_per_fold': Array of AUC values for each fold
    """
    y = data["y"]
    y_score = data["y_score"]
    fold_indices = data["fold_indices"]
    N = data["N"]

    k = len(np.unique(fold_indices))
    Nk = N // k

    # Compute pooled AUC (entire vector)
    auc_pooled = roc_auc_score(y, y_score)

    # Compute AUC per fold
    auc_per_fold = np.zeros(k)
    for fold in range(k):
        start_idx = fold * Nk
        end_idx = start_idx + Nk

        fold_y = y[start_idx:end_idx]
        fold_scores = y_score[start_idx:end_idx]

        # Check if fold has both classes (needed for AUC calculation)
        if len(np.unique(fold_y)) > 1:
            auc_per_fold[fold] = roc_auc_score(fold_y, fold_scores)
        else:
            auc_per_fold[fold] = np.nan

    # Compute weighted average of per-fold AUCs
    # Weight = Nk / N (each fold has equal weight if all folds have same size)
    weight = Nk / N
    auc_weighted = np.nansum(auc_per_fold * weight)

    return {"auc_pooled": auc_pooled, "auc_weighted": auc_weighted, "auc_per_fold": auc_per_fold}


def simulate_cv_data(
    k=5,
    N=500,
    positive_ratio=0.3,
    pos_score_range=(0.4, 0.95),
    neg_score_range=(0.05, 0.6),
    case="single_overlap",
    random_seed=None,
):
    """
    Simulate cross-validation labels and scores with perfect classification
    in first k-1 folds and imperfect classification in the last fold.

    Parameters
    ----------
    k : int, default=5
        Number of folds
    N : int, default=500
        Total number of samples
    positive_ratio : float, default=0.3
        Ratio of positive samples per fold (between 0 and 1)
    pos_score_range : tuple, default=(0.4, 0.95)
        Range of scores for positive class in the last fold (min, max)
    neg_score_range : tuple, default=(0.05, 0.6)
        Range of scores for negative class in the last fold (min, max)
    case : str, default="single_overlap"
        Case for last fold: "single_overlap", "two_overlap", or "multiple_overlap"
    random_seed : int or None, default=None
        Random seed for reproducibility

    Returns
    -------
    dict
        Dictionary containing:
        - 'y': array of labels (0 or 1)
        - 'y_score': array of predicted scores [0, 1]
        - 'fold_indices': array indicating which fold each sample belongs to
        - 'N': total number of samples
        - 'k': number of folds
        - 'Nk': number of samples per fold
        - 'Npk': number of positives per fold
        - 'Nnk': number of negatives per fold
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if not 0 <= positive_ratio <= 1:
        raise ValueError("positive_ratio must be between 0 and 1")

    if N % k != 0:
        raise ValueError(f"N ({N}) must be divisible by k ({k})")

    Nk = N // k  # number of samples per fold
    Npk = int(Nk * positive_ratio)  # number of positives per fold
    Nnk = Nk - Npk  # number of negatives per fold

    # Initialize arrays
    y = np.zeros(N, dtype=int)
    y_score = np.zeros(N)  # predicted scores of belonging to class 0
    fold_indices = np.zeros(N, dtype=int)

    # Generate data for each fold
    for fold in range(k):
        start_idx = fold * Nk
        end_idx = start_idx + Nk

        # Assign fold indices
        fold_indices[start_idx:end_idx] = fold

        # Create labels: Npk positives (1) and Nnk negatives (0)
        y[start_idx : start_idx + Npk] = 1
        y[start_idx + Npk : end_idx] = 0

        if fold < k - 1:
            # Perfect classification for first k-1 folds
            # For some weird reason, sckit-learn wants the probaility for being 0, not 1. Therefore we "invert" the scores
            y_score[start_idx : start_idx + Npk] = np.random.uniform(0.0, 0.2, size=Npk)
            y_score[start_idx + Npk : end_idx] = np.random.uniform(0.8, 1.0, size=Nnk)
        else:
            if case == "single_overlap":  # only one overlapping
                y_score[start_idx : start_idx + Npk] = np.random.uniform(0.0, 0.2, size=Npk)
                y_score[start_idx + Npk : end_idx] = np.random.uniform(0.8, 1.0, size=Nnk)
                y_score[end_idx - 1] = 0.4
                y_score[start_idx + Npk - 1] = 0.6
            elif case == "two_overlap":  # two overlapping
                y_score[start_idx : start_idx + Npk] = np.random.uniform(0.0, 0.2, size=Npk)
                y_score[start_idx + Npk : end_idx] = np.random.uniform(0.8, 1.0, size=Nnk)
                y_score[end_idx - 1] = 0.4
                y_score[end_idx - 2] = 0.45
                y_score[start_idx + Npk - 1] = 0.6
                y_score[start_idx + Npk - 2] = 0.55

            elif case == "multiple_overlap":  # multiple overlapping
                y_score[start_idx : start_idx + Npk] = np.random.uniform(0.0, 0.6, size=Npk)
                y_score[start_idx + Npk : end_idx] = np.random.uniform(0.4, 1.0, size=Nnk)

    return {"y": y, "y_score": y_score, "fold_indices": fold_indices, "N": N, "k": k, "Nk": Nk, "Npk": Npk, "Nnk": Nnk}


def print_cv_statistics(data):
    """
    Print summary statistics for simulated cross-validation data.

    Parameters
    ----------
    data : dict
        Output from simulate_cv_data()
    """
    y = data["y"]
    y_score = data["y_score"]
    fold_indices = data["fold_indices"]
    N = data["N"]

    k = len(np.unique(fold_indices))
    Nk = N // k
    Npk = data["Npk"]
    Nnk = data["Nnk"]

    print(f"Total samples: {N}")
    print(f"Number of folds: {k}")
    print(f"Samples per fold: {Nk}")
    print(f"Positives per fold: {Npk}")
    print(f"Negatives per fold: {Nnk}")
    print(f"\nLabel distribution: {np.bincount(y)}")

    # Print statistics per fold
    print("\nPer-fold statistics:")
    for fold in range(k):
        start_idx = fold * Nk
        end_idx = start_idx + Nk

        fold_y = y[start_idx:end_idx]
        fold_scores = y_score[start_idx:end_idx]

        # Calculate accuracy (assuming threshold = 0.5)
        predictions = (fold_scores >= 0.5).astype(int)
        accuracy = np.mean(predictions == fold_y)

        print(
            f"Fold {fold}: Accuracy = {accuracy:.3f}, "
            f"Mean score = {fold_scores.mean():.3f}, "
            f"Positive rate = {fold_y.mean():.3f}"
        )


def plot_cv_data(data):
    """
    Visualize simulated cross-validation data.

    Parameters
    ----------
    data : dict
        Output from simulate_cv_data()
    """
    y = data["y"]
    y_score = data["y_score"]
    fold_indices = data["fold_indices"]
    N = data["N"]

    k = len(np.unique(fold_indices))
    Nk = N // k

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Score distribution by label for each fold
    ax1 = axes[0]
    for fold in range(k):
        start_idx = fold * Nk
        end_idx = start_idx + Nk

        fold_y = y[start_idx:end_idx]
        fold_scores = y_score[start_idx:end_idx]

        pos_scores = fold_scores[fold_y == 1]
        neg_scores = fold_scores[fold_y == 0]

        offset = fold * 0.15
        ax1.scatter(
            [fold + offset] * len(pos_scores),
            pos_scores,
            alpha=0.3,
            c="red",
            s=20,
            label=f"Fold {fold} Positive" if fold == 0 else "",
        )
        ax1.scatter(
            [fold + offset] * len(neg_scores),
            neg_scores,
            alpha=0.3,
            c="blue",
            s=20,
            label=f"Fold {fold} Negative" if fold == 0 else "",
        )

    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Score")
    ax1.set_title("Score Distribution by Fold and Label")
    ax1.axvline(k - 1.5, color="green", linestyle="--", linewidth=2, label="Last fold boundary")
    ax1.legend(["Positive", "Negative", "Last fold boundary"])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of scores for last fold
    ax2 = axes[1]
    last_fold_start = (k - 1) * Nk
    last_fold_y = y[last_fold_start:]
    last_fold_scores = y_score[last_fold_start:]

    ax2.hist(last_fold_scores[last_fold_y == 1], bins=20, alpha=0.6, color="red", label="Positive (y=1)")
    ax2.hist(last_fold_scores[last_fold_y == 0], bins=20, alpha=0.6, color="blue", label="Negative (y=0)")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Decision threshold")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.set_title(f"Score Distribution in Last Fold (Fold {k-1})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def calculate_theoretical_difference(N, K, pos_ratio, Nbn=1, Nbp=1):
    Np = N * pos_ratio
    Nn = N * (1 - pos_ratio)
    return (K - 1) * (Nbn * Nbp) / (Np * Nn)


def run_simulation(Ks, N, positive_ratios, case="single_overlap"):
    print(f"Running simulation for case: {case}")
    print(f"N={N}, Ks={Ks}, positive_ratios={positive_ratios}")
    results_matrix = np.zeros((len(Ks), len(positive_ratios)))
    results_matrix_theoretical = np.zeros((len(Ks), len(positive_ratios)))

    for i, K in enumerate(Ks):
        for j, positive_ratio in enumerate(positive_ratios):
            data = simulate_cv_data(k=K, N=N, positive_ratio=positive_ratio, case=case, random_seed=123)

            # Compute AUC metrics
            auc_results = compute_auc(data)

            # difference between pooled and weighted
            diff = auc_results["auc_weighted"] - auc_results["auc_pooled"]

            # Theoretical difference
            theoretical_diff = calculate_theoretical_difference(N, K, positive_ratio)

            # Store results
            results_matrix[i, j] = diff
            results_matrix_theoretical[i, j] = theoretical_diff

    return results_matrix, results_matrix_theoretical


# Example usage
if __name__ == "__main__":

    ## simulation parameters:
    Ks = [2, 5, 10]
    N = 100
    positive_ratios = np.linspace(0.2, 0.8, 7)

    ## Case 1: 1 missclassification in last fold
    results_matrix, results_matrix_theoretical = run_simulation(Ks, N, positive_ratios, case="single_overlap")

    plt.figure(figsize=(8, 6))
    for i, K in enumerate(Ks):
        plt.plot(positive_ratios, results_matrix[i], marker="o", label=f"K={K} (Simulated)")
        plt.plot(
            positive_ratios, results_matrix_theoretical[i], marker="x", linestyle="--", label=f"K={K} (Theoretical)"
        )
    plt.xlabel("Positive class ratio")
    plt.ylabel("Difference between averaged and pooled AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot example data
    example_data = simulate_cv_data(k=5, N=100, positive_ratio=0.3, case="single_overlap", random_seed=42)
    plot_cv_data(example_data)

    ## Case 2: 2 missclassifications in last fold
    results_matrix, results_matrix_theoretical = run_simulation(Ks, N, positive_ratios, case="two_overlap")
    plt.figure(figsize=(8, 6))
    for i, K in enumerate(Ks):
        plt.plot(positive_ratios, results_matrix[i], marker="o", label=f"K={K} (Simulated)")
        plt.plot(
            positive_ratios, results_matrix_theoretical[i], marker="x", linestyle="--", label=f"K={K} (Theoretical)"
        )
    plt.xlabel("Positive class ratio")
    plt.ylabel("Difference between averaged and pooled AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot example data
    example_data = simulate_cv_data(k=5, N=100, positive_ratio=0.3, case="two_overlap", random_seed=42)
    plot_cv_data(example_data)

    ## Case 3: multiple missclassifications in last fold
    results_matrix, results_matrix_theoretical = run_simulation(Ks, N, positive_ratios, case="multiple_overlap")
    plt.figure(figsize=(8, 6))
    for i, K in enumerate(Ks):
        plt.plot(positive_ratios, results_matrix[i], marker="o", label=f"K={K} (Simulated)")
        plt.plot(
            positive_ratios, results_matrix_theoretical[i], marker="x", linestyle="--", label=f"K={K} (Theoretical)"
        )
    plt.xlabel("Positive class ratio")
    plt.ylabel("Difference between averaged and pooled AUC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Plot example data
    example_data = simulate_cv_data(k=5, N=100, positive_ratio=0.3, case="multiple_overlap", random_seed=42)
    plot_cv_data(example_data)

# %%
