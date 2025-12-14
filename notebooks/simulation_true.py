#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.formula.api as smf
import pingouin as pg
from src.cv import run_cv
from src import simulations
from src.metrics import create_metrics
import seaborn as sns

#%% functions
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
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=False)
    return fig, axes


#%% Simulate data
# set up the dataset parameters
n_samples = 500
pos_ratio = 0.1
mu0 = -2
sigma0 = 1
mu1 = 2
sigma1 = 1
seed = 42
# simulate
X, y = simulations.simulate_dataset(n_samples, pos_ratio, mu0, sigma0, mu1, sigma1, seed=seed)

# plot the dataset
fig, ax = plt.subplots()
ax.set_title(f"Simulated dataset (pos_ratio={pos_ratio}, n_samples={n_samples})")
ax.set_xlabel("Feature value")
ax.set_ylabel("Count")
ax.hist(X[y == 0], bins=30, alpha=0.5, label="Class 0", color="blue")
ax.hist(X[y == 1], bins=30, alpha=0.5, label="Class 1", color="orange")
ax.legend()
plt.show()

#%% sample a large dataset to get the "True" prediction error
posratios = np.linspace(0.1, 0.9, 9)

results_all = []
for i, pos_ratio in enumerate(posratios):
    results = []

    for seed in range(1, 101):
        # simulate new set of data
        X, y = simulations.simulate_dataset(n_samples, pos_ratio, mu0, sigma0, mu1, sigma1, seed=i*1000 + seed)

        # setup model
        model = LogisticRegression()
        metrics = create_metrics(["accuracy", "rocauc", "prcauc"])

        # run cv
        result = run_cv(model, X, y, metrics, n_splits=5, stratified=True, random_state=1)

        # combine
        df_average = pd.DataFrame([result["average"]])
        df_average["seed"] = seed
        df_average["method"] = "average"
        df_pooled = pd.DataFrame([result["pooled"]])
        df_pooled["seed"] = seed
        df_pooled["method"] = "pooled"
        df_result = pd.concat([df_average, df_pooled], ignore_index=True)
        results.append(df_result)

    results_df = pd.concat(results, ignore_index=True)


    # get "true" values from large dataset
    X_large, y_large = simulations.simulate_dataset(100000, pos_ratio, mu0, sigma0, mu1, sigma1, seed=i*1000 + seed)
    model.fit(X_large, y_large)
    y_pred_large = model.predict_proba(X_large)[:, 1]
    true_rocauc = metrics["rocauc"](y_large, None, y_pred_large)
    true_prcauc = metrics["prcauc"](y_large, None, y_pred_large)
    true_accuracy = metrics["accuracy"](y_large, (y_pred_large >= 0.5).astype(int), None)

    true_result = {"rocauc": true_rocauc,
                    "prcauc": true_prcauc,
                    "accuracy": true_accuracy}
    df_true = pd.DataFrame([true_result])
    df_true["method"] = "true"
    results_df = pd.concat([results_df, df_true], ignore_index=True)

    results_df["pos_ratio"] = pos_ratio
    results_all.append(results_df)

results_all = pd.concat(results_all, ignore_index=True)

#%% 
fig, axes = setup_plotting()
sns.lineplot(x="pos_ratio", y="rocauc", hue= "method", data=results_all, ax=axes[0], estimator='mean', errorbar=('ci', 95))
axes[0].set_ylabel("AURCO")
axes[0].set_xlabel("Positive Class Ratio")
    
sns.lineplot(x="pos_ratio", y="prcauc", hue= "method", data=results_all, ax=axes[1], estimator='mean', errorbar=('ci', 95))
axes[1].set_ylabel("AUPRC")
axes[1].set_xlabel("Positive Class Ratio")
fig.tight_layout()
fig.savefig("rocauc_vs_pos_ratio.png")
# %%
