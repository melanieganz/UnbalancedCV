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
    fig_width = 17 * cm
    fig_height = 17 * cm / 3
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=False)
    return fig, axes


def prep_breast_cancer():
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
    return X, y

def prep_cognitive_impairment():
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
    return X_scaled, y_binary


def prep_depression_remission():
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
    y_binary = (y == "Recurrent").astype(bool)

    # Check class balance
    print(
        f"Class distribution:\n{sum(y_binary)} positive, {len(y_binary) - sum(y_binary)} negative, ratio: {sum(y_binary) / len(y_binary):.2f}"
    )
    return X_scaled, y_binary

def run_experiment(X, y, ax, flipped: bool = False, metric: str = "rocauc"):
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

    # Paired t-test (using statsmodels)
    diff = np.array(avg_rocauc) - np.array(pooled_rocauc)
    model_paired = smf.ols(formula="diff ~ 1", data=pd.DataFrame({"diff": diff})).fit()
    coef_diff = model_paired.params['Intercept']
    ci_lower = model_paired.conf_int().loc['Intercept'][0]
    ci_upper = model_paired.conf_int().loc['Intercept'][1]
    p_value = model_paired.pvalues['Intercept']

    cohens_d = pg.compute_effsize(avg_rocauc, pooled_rocauc, paired=True, eftype='cohen')

    stats = {"average_mean": avg_mean,
             "pooled_mean": pooled_mean,
             "diff": coef_diff,
             "95_ci_lower": ci_lower,
             "95_ci_upper": ci_upper,
             "p_value": p_value,
             "cohens_d": cohens_d,
             "N": len(y),
             "positive_class_ratio": sum(y) / len(y)}

    # Plotting
    # boxplot with scatter points
    ax.scatter([0]*len(avg_rocauc), avg_rocauc, color='blue', alpha=0.6, label=f'Average {metric.capitalize()}', s=4)
    ax.scatter([1]*len(pooled_rocauc), pooled_rocauc, color='orange', alpha=0.6, label=f'Pooled {metric.capitalize()}', s=4)

    # plot black point for mean with error bars for 95% CI
    ax.errorbar(0, avg_mean, yerr=[[avg_mean - avg_ci_lower], [avg_ci_upper - avg_mean]], fmt='o', color='black', capsize=3, linewidth=1, markersize=4)
    ax.errorbar(1, pooled_mean, yerr=[[pooled_mean - pooled_ci_lower], [pooled_ci_upper - pooled_mean]], fmt='o', color='black', capsize=3, linewidth=1, markersize=4)    

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Average', 'Pooled'])
    ax.set_ylabel(metric.capitalize())
    ax.set_xlim(-0.5, 1.5)

    return stats


def run_experiment_metric(metric: str):
    """ could be rocauc or prcauc """
    fig, axes = setup_plotting()

    # breast cancer
    X_breast_cancer, y_breast_cancer = prep_breast_cancer()
    stats_breast_cancer = run_experiment(X_breast_cancer, y_breast_cancer, axes[0], metric=metric)
    axes[0].set_xlabel('a)')

    # cognitive impairment
    X_cognitive_impairment, y_cognitive_impairment = prep_cognitive_impairment()
    stats_cognitive_impairment = run_experiment(X_cognitive_impairment, y_cognitive_impairment, axes[1], metric=metric)
    axes[1].set_xlabel('b)')

    # depression remission
    X_depression_remission, y_depression_remission = prep_depression_remission()
    stats_depression_remission = run_experiment(X_depression_remission, y_depression_remission, axes[2], flipped=True, metric=metric)
    axes[2].set_xlabel('c)')

    # plot finalization
    fig.tight_layout()
    plt.show()
    fig.savefig(f"example_repeated__{metric}_cv.png")
    fig.savefig(f"example_repeated_{metric}_cv.pdf")

    # stats output (make one table to be saved as tsv)
    stats = pd.DataFrame({
        "Breast Cancer": stats_breast_cancer,
        "Cognitive Impairment": stats_cognitive_impairment,
        "Depression Remission": stats_depression_remission
    })


    # round to 5 decimal places
    stats = stats.round(5)
    # transpose the dataframe
    stats = stats.T
    stats.to_csv(f"example_repeated_{metric}_cv_stats.csv", sep=",")

    
if __name__ == "__main__":
    run_experiment_metric("rocauc")
    run_experiment_metric("prcauc")


