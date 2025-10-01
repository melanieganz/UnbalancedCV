# %%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.cv import run_cv
from src import simulations
from src.metrics import create_metrics
import numpy as np

# set up the dataset parameters
n_samples = 100
mu0 = -1
sigma0 = 1
mu1 = 1
sigma1 = 1
pos_ratios = np.linspace(0.2, 0.8, 10)
K_values = [3, 5, 7, 10]

results_matrix = np.zeros((len(K_values), len(pos_ratios)))

for i, K in enumerate(K_values):
    diff_aucs = []
    for j, pos_ratio in enumerate(pos_ratios):
        X, y = simulations.simulate_dataset(n_samples, pos_ratio, mu0, sigma0, mu1, sigma1, seed=42)
        model = LogisticRegression()
        metrics = create_metrics(["accuracy", "rocauc", "prcauc"])
        results = run_cv(model, X, y, metrics, n_splits=K, stratified=True, random_state=42)
        df_results = pd.DataFrame(results)
        diff_auc = df_results["pooled"]["rocauc"] - df_results["average"]["rocauc"]
        diff_aucs.append(diff_auc)
        results_matrix[i, j] = diff_auc

plt.figure(figsize=(8, 6))
for i, K in enumerate(K_values):
    plt.plot(pos_ratios, results_matrix[i], marker="o", label=f"K={K}")
plt.xlabel("Positive class ratio")
plt.ylabel("Difference between pooled and weighted AUC")
plt.title("AUC Difference vs. Positive Class Ratio for different K")
plt.legend(title="Number of Folds (K)")
plt.grid(True)
plt.savefig("advExamplesResults.png")

# %%
