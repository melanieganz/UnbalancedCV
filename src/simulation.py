# %% 1. Simulate Data from Class-Conditional Distributions
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# %% Set seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000
pos_ratio = 0.1  # class imbalance: 20% positives

# Define class-conditional distributions
mu_0, sigma_0 = -1, 1
mu_1, sigma_1 = 1, 1

# Sample class labels: y ~ Bernoulli(pos_ratio)
y = np.random.binomial(1, pos_ratio, size=n_samples)

# Sample x ~ P(x | y)
x = np.where(y == 0, np.random.normal(mu_0, sigma_0, size=n_samples), np.random.normal(mu_1, sigma_1, size=n_samples))

# Reshape x for sklearn
X = x.reshape(-1, 1)

# %%
plt.hist(x[y == 0], bins=30, alpha=0.6, label="y = 0")
plt.hist(x[y == 1], bins=30, alpha=0.6, label="y = 1")
plt.xlabel("x")
plt.ylabel("Count")
plt.title("Class-conditional distributions of x")
plt.legend()
plt.show()


# %% function to compute the accuracy
def calculate_accuracy(predictions, targets):
    """
    Calculates the accuracy of the model's predictions.

    Args:
        predictions (torch.Tensor): Model predictions (e.g., logits or probabilities)
        targets (torch.Tensor): Ground truth labels

    Returns:
        float: Accuracy value between 0 and 1
    """
    # Check where predictions match targets
    correct_predictions = sum(predictions == targets)

    # Calculate accuracy
    accuracy = correct_predictions / len(targets)

    return accuracy


# %%
clf = LogisticRegression(solver="lbfgs")

cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Iterate through cv and save predictions
predictions = []
true_labels = []
cv_error = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    predictions.extend(y_pred)
    true_labels.extend(y_test)

    # compute the loss per cv
    fold_error = calculate_accuracy(y_pred, y_test)
    cv_error.append(fold_error)

# %% compute global and local loss
# Compute global accuracy
global_auc = calculate_accuracy(np.array(predictions), np.array(true_labels))
print(f"Global AUC: {global_auc}")

# Compute local loss (mean of cv_error)
local_loss = np.mean(cv_error)
print(f"Local Loss (Mean AUC across folds): {local_loss}")

# %%
