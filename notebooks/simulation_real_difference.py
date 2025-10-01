import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def simulate_cv_data_random(
    k=5, N=500, positive_ratio=0.3, pos_score_range=(0.4, 0.95), neg_score_range=(0.05, 0.6), random_seed=None
):
    """
    Simulate cross-validation labels and scores with random classification
    in all folds (with overlap between positive and negative score distributions).

    Parameters
    ----------
    k : int, default=5
        Number of folds
    N : int, default=500
        Total number of samples
    positive_ratio : float, default=0.3
        Ratio of positive samples per fold (between 0 and 1)
    pos_score_range : tuple, default=(0.4, 0.95)
        Range of scores for positive class in all folds (min, max)
    neg_score_range : tuple, default=(0.05, 0.6)
        Range of scores for negative class in all folds (min, max)
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
    y_score = np.zeros(N)
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

        # Random/imperfect classification for ALL folds
        # Positives get scores from pos_score_range
        y_score[start_idx : start_idx + Npk] = np.random.uniform(pos_score_range[0], pos_score_range[1], Npk)
        # Negatives get scores from neg_score_range
        y_score[start_idx + Npk : end_idx] = np.random.uniform(neg_score_range[0], neg_score_range[1], Nnk)

    # Shuffle within each fold to mix positives and negatives
    for fold in range(k):
        start_idx = fold * Nk
        end_idx = start_idx + Nk

        shuffle_idx = np.random.permutation(Nk) + start_idx
        y[start_idx:end_idx] = y[shuffle_idx]
        y_score[start_idx:end_idx] = y_score[shuffle_idx]

    return {"y": y, "y_score": y_score, "fold_indices": fold_indices, "N": N, "k": k, "Nk": Nk, "Npk": Npk, "Nnk": Nnk}


def compute_auc(data):
    """
    Compute AUC on pooled data and as weighted average across folds.

    Parameters
    ----------
    data : dict
        Output from simulate_cv_data_random()

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


def print_cv_statistics(data):
    """
    Print summary statistics for simulated cross-validation data.

    Parameters
    ----------
    data : dict
        Output from simulate_cv_data_random()
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
        Output from simulate_cv_data_random()
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
    ax1.legend(["Positive", "Negative"])
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of scores for all folds combined
    ax2 = axes[1]

    ax2.hist(y_score[y == 1], bins=20, alpha=0.6, color="red", label="Positive (y=1)")
    ax2.hist(y_score[y == 0], bins=20, alpha=0.6, color="blue", label="Negative (y=0)")
    ax2.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Decision threshold")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution (All Folds)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    pos_ratios = np.linspace(0.1, 0.9, 9)
    k_values = [2, 5, 10]
    diffs_matrix = np.zeros((len(k_values), len(pos_ratios)))
    auc_pooled_matrix = np.zeros_like(diffs_matrix)
    auc_weighted_matrix = np.zeros_like(diffs_matrix)

    for i, k in enumerate(k_values):
        for j, pos_ratio in enumerate(pos_ratios):
            data = simulate_cv_data_random(k=k, N=500, positive_ratio=pos_ratio, random_seed=42)
            auc_results = compute_auc(data)
            diff = auc_results["auc_pooled"] - auc_results["auc_weighted"]
            diffs_matrix[i, j] = diff
            auc_pooled_matrix[i, j] = auc_results["auc_pooled"]
            auc_weighted_matrix[i, j] = auc_results["auc_weighted"]

    # Plot difference as a function of positive ratio for each k
    plt.figure(figsize=(8, 5))
    for i, k in enumerate(k_values):
        plt.plot(pos_ratios, diffs_matrix[i], marker="o", label=f"K={k}")
    plt.xlabel("Positive Ratio")
    plt.ylabel("Difference (Pooled - Weighted AUC)")
    plt.title("Difference between Pooled and Weighted AUC vs Positive Ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
