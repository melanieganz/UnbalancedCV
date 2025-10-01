import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
k = 5  # number of folds
Nk = 100  # number of elements per fold
N = k * Nk  # total number of elements
Npk = 30  # number of positives per fold
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

    if fold < k - 1:
        # Perfect classification for first k-1 folds
        # Positives get score = 1, negatives get score = 0
        y_score[start_idx : start_idx + Npk] = 1.0
        y_score[start_idx + Npk : end_idx] = 0.0
    else:
        # Imperfect classification for last fold
        # Add some noise/errors
        # Positives get scores around 0.7 with noise
        y_score[start_idx : start_idx + Npk] = np.random.uniform(0.4, 0.95, Npk)
        # Negatives get scores around 0.3 with noise
        y_score[start_idx + Npk : end_idx] = np.random.uniform(0.05, 0.6, Nnk)

# Shuffle within each fold to mix positives and negatives
for fold in range(k):
    start_idx = fold * Nk
    end_idx = start_idx + Nk

    shuffle_idx = np.random.permutation(Nk) + start_idx
    y[start_idx:end_idx] = y[shuffle_idx]
    y_score[start_idx:end_idx] = y_score[shuffle_idx]

# Print summary statistics
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

# Visualization
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

# Export the data
print("\nData shapes:")
print(f"y shape: {y.shape}")
print(f"y_score shape: {y_score.shape}")
print(f"fold_indices shape: {fold_indices.shape}")
