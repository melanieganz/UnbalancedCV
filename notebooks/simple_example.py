# %%
# import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, PrecisionRecallDisplay
import numpy as np
import matplotlib.pyplot as plt

k = 2
Nk = 5
N = k * Nk

# Example 2: Simple binary classification
y_true = np.array([1, 0, 1, 0, 0, 1, 1, 1, 1, 0])
y_scores = np.array([0.8, 0.2, 0.7, 0.4, 0.45, 0.55, 0.35, 0.7, 0.8, 0.55])

print("y_true:", y_true)
print("y_scor:", (y_scores > 0.5).astype(int))

# chuck it into k folds
folds = [np.arange(i * Nk, (i + 1) * Nk) for i in range(k)]
folds_aucr_cor = []
for i, fold in enumerate(folds):
    print(f"Fold {i+1}:")
    # compute AUC-ROC for each fold
    fold_auc_roc = roc_auc_score(y_true[fold], y_scores[fold])
    print(f"  AUC-ROC: {fold_auc_roc:.4f}")
    folds_aucr_cor.append(fold_auc_roc)

# compute the pooled AUC-ROC
pooled_auc_roc = roc_auc_score(y_true, y_scores)

# average AUC-ROC across folds
avg_auc_roc = np.nansum(Nk / N * np.array(folds_aucr_cor))

print(f"Average AUC-ROC across folds: {avg_auc_roc:.4f}")
print(f"Pooled AUC-ROC: {pooled_auc_roc:.4f}")

theoretical_diff = (k - 1) / (sum(y_true == 1) * sum(y_true == 0))
experimental_diff = pooled_auc_roc - avg_auc_roc

print(f"Theoretical difference: {theoretical_diff:.6f}")
print(f"Experimental difference: {experimental_diff:.6f}")


# %%
def calculate_theoretical_difference(N, K, pos_ratio):
    Np = N * pos_ratio
    Nn = N * (1 - pos_ratio)
    return (K - 1) / (Np * Nn)


# %%
N = 100
pos_ratios = np.linspace(0.01, 1, 100)
k = np.array([3, 5, 7, 10])


plt.figure(figsize=(10, 6))
for i, K in enumerate(k):
    theoretical_diffs = calculate_theoretical_difference(N, K, pos_ratios)
    plt.plot(pos_ratios, theoretical_diffs, label=f"K={K}")

plt.xlabel("Positive Class Ratio")
plt.ylabel("Theoretical Difference")
plt.title("Theoretical Difference vs Positive Class Ratio")
plt.legend()
plt.grid()
plt.savefig("theoretical_difference_vs_pos_ratio.png")

# %% Simulations for this case:

# %%
