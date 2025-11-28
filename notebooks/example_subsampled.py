from pathlib import Path

import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import our custom functions
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.metrics import create_metrics
from src.cv import run_repeated_cv


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
    fig_width = 17 * cm /3
    fig_height = 17 * cm / 3
    fig, axes = plt.subplots(1, 1, figsize=(fig_width, fig_height), sharey=False)
    return fig, axes


def prep_breast_cancer(sample_percentage, seed=1):
    """
    Example 1: Breast Cancer Prediction
    ====================================
    Classification task: predict whether the subject has cancer or not.
    """
    X, y = load_breast_cancer(return_X_y=True)

    # randomly subsample so that the dataset contains sample_percentage percent of the total data
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_subsample = int(n_samples * sample_percentage / 100)
    indices = np.random.choice(n_samples, n_subsample, replace=False)

    X_subsapled = X[indices]
    y_subsampled = y[indices]

    # print dataset shape
    print(f"Dataset shape: {X_subsapled.shape}")
    # Check class balance
    print(f"Class distribution:\n{sum(y_subsampled)} positive, {len(y_subsampled) - sum(y_subsampled)} negative, ratio: {sum(y_subsampled) / len(y_subsampled):.2f}")
    return X_subsapled, y_subsampled
    

def run_experiment(X, y, flipped: bool = False, metric: str = "rocauc"):
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run cross-validation
    model = LogisticRegression()
    metrics = create_metrics(["accuracy", "rocauc", "prcauc"])
    results = run_repeated_cv(model, 
                                X_scaled, 
                                y, 
                                metrics, 
                                n_splits=5, 
                                n_repeats=100,
                                stratified=True, 
                                random_state=1,
                                flipped=flipped)

    # Plot scatter plot of average and pooled AUC have AUC on the y-axis and average or pooled on the x-axis
    avg_rocauc = [res[metric] for res in results["average"]]
    pooled_rocauc = [res[metric] for res in results["pooled"]]

    # convert it into a long dataframe with column type and average = 0, pooled = 1
    df = pd.DataFrame({
        metric: avg_rocauc + pooled_rocauc,
        "type": ["average"] * len(avg_rocauc) + ["pooled"] * len(pooled_rocauc),
    })

    # stats
    avg_mean = np.mean(avg_rocauc)
    avg_se = np.std(avg_rocauc) / (len(avg_rocauc) ** 0.5)
    avg_ci_upper = avg_mean + 1.96 * avg_se
    avg_ci_lower = avg_mean - 1.96 * avg_se

    pooled_mean = np.mean(pooled_rocauc)
    pooled_se = np.std(pooled_rocauc) / (len(pooled_rocauc) ** 0.5)
    pooled_ci_upper = pooled_mean + 1.96 * pooled_se
    pooled_ci_lower = pooled_mean - 1.96 * pooled_se

    print(f"Average {metric} over repeats: {avg_mean:.4f}, 95% CI: [{avg_ci_lower:.4f}, {avg_ci_upper:.4f}]")
    print(f"Pooled {metric} over repeats: {pooled_mean:.4f}, 95% CI: [{pooled_ci_lower:.4f}, {pooled_ci_upper:.4f}]")

    # statsmodels for testing (use the formula API)
    model = smf.ols(formula=f"{metric} ~ C(type)", data=df).fit()
    coef_diff = model.params['C(type)[T.pooled]']
    residual_sd = np.sqrt(model.mse_resid)
    cohens_d_sm = coef_diff / residual_sd

    print(f"Difference in pooled {metric} (statsmodels): coef={model.params['C(type)[T.pooled]']:.4f} 95%CI[{model.conf_int().loc['C(type)[T.pooled]'][0]:.4f}, {model.conf_int().loc['C(type)[T.pooled]'][1]:.4f}], p={model.pvalues['C(type)[T.pooled]']:.4f}, Cohen's d={cohens_d_sm:.4f}")
    print(avg_mean - pooled_mean)

    stats = {"average_mean": avg_mean,
             "pooled_mean": pooled_mean,
             "diff": coef_diff,
             "95_ci_lower": model.conf_int().loc['C(type)[T.pooled]'][0],
             "95_ci_upper": model.conf_int().loc['C(type)[T.pooled]'][1],
             "p_value": model.pvalues['C(type)[T.pooled]'],
             "cohens_d": cohens_d_sm,
             "N": len(y),
             "positive_class_ratio": sum(y) / len(y)}

    return stats


if __name__ == "__main__":
    fig, axes = setup_plotting()

    subsample_percentages = [100, 90, 80, 70, 60, 50]
    results = []
    for percentage in subsample_percentages:
        X_subsampled, y_subsampled = prep_breast_cancer(percentage, 1)
        stats = run_experiment(X_subsampled, y_subsampled, metric="rocauc")
        print(stats)
        results.append(stats)

    df = pd.DataFrame(results)
    print(df)
    lower = df["diff"] - df["95_ci_lower"]
    upper = df["95_ci_upper"] - df["diff"]
    yerr = np.vstack([lower.to_numpy(), upper.to_numpy()])  # shape (2, N)
    axes.errorbar(
        subsample_percentages,
        df["diff"],
        yerr=yerr,
        fmt='o',
        color='black',
        capsize=3,
        linewidth=1,
        markersize=4,
    )

    axes.set_xticks(subsample_percentages)
    axes.set_xlabel("Dataset Size in Percent")
    axes.set_ylabel("Difference Pooled - Averaged")
    fig.tight_layout()
    fig.savefig(f"example_subsampled_rocauc_cv.png")
    plt.show()
