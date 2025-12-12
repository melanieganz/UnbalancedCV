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
from ucimlrepo import fetch_ucirepo 

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
    fig, axes = plt.subplots(1, 4, figsize=(fig_width, fig_height), sharey=False)
    return fig, axes

def load_heart_failure_data():
    # fetch dataset 
    heart_failure_clinical_records = fetch_ucirepo(id=519) 
    
    # data (as pandas dataframes) 
    X = heart_failure_clinical_records.data.features
    X = X.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)
    X = X.to_numpy()

    y = heart_failure_clinical_records.data.targets.to_numpy().ravel()

    # convert y to int or bool
    y = (y == 1).astype(int)

    # remove nans
    X, y = drop_nans(X,y)

    return X, y


def load_obesity_data(): 
    # fetch dataset 
    estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544) 
    
    # data (as pandas dataframes) 
    X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
   
    X = X.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)
    X = X.to_numpy()

    y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets.to_numpy().ravel()

    # only keep 'Normal_Weight' and 'Overweight_Level_II'
    keep = np.isin(y, ['Normal_Weight', 'Overweight_Level_II'])
    
    X = X[keep]
    y = y[keep]

    # convert y to int or bool
    y = (y == 'Overweight_Level_II').astype(int)

    # remove nans
    X, y = drop_nans(X,y)

    return X, y


def load_heart_disease_data():
    # fetch dataset 
    heart_disease = fetch_ucirepo(id=45) 
    
    # data (as pandas dataframes) 
    X = heart_disease.data.features
    # make sure that categories are converted to integers
    X = X.apply(lambda col: col.astype('category').cat.codes if col.dtype == 'object' else col)
    X = X.to_numpy()
    y = heart_disease.data.targets.to_numpy().ravel()

    # only keep level 0 and 4
    keep = np.isin(y, [0,4])
    X = X[keep]
    y = y[keep]

    # convert to bool
    y = (y==4).astype(int)

    # remove nans
    X, y = drop_nans(X,y)

    return X, y


def load_cognitive_impairment():
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


def load_depression_remission():
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


def subsample_dataset(X, y, sample_percentage, seed=1):
    """Randomly subsample the dataset to the given percentage."""
    np.random.seed(seed)
    n_samples = X.shape[0]
    n_subsample = int(n_samples * sample_percentage / 100)
    indices = np.random.choice(n_samples, n_subsample, replace=False)

    X_subsampled = X[indices]
    y_subsampled = y[indices]

    # print dataset shape
    print(f"Dataset shape: {X_subsampled.shape}")
    # Check class balance
    print(f"Class distribution:\n{sum(y_subsampled)} positive, {len(y_subsampled) - sum(y_subsampled)} negative, ratio: {sum(y_subsampled) / len(y_subsampled):.2f}")

    return X_subsampled, y_subsampled


def drop_nans (X,y): 
    # funtion to drop rows with NaNs in X. should remove the coresponding entries in y as well.
    not_nan = ~np.isnan(X).any(axis=1)

    X_clean = X[not_nan]
    y_clean = y[not_nan]
    return X_clean, y_clean


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

    avg_mean = np.mean(avg_rocauc)
    pooled_mean = np.mean(pooled_rocauc)

    # paired t-test
    diff = np.array(avg_rocauc) - np.array(pooled_rocauc)
    model = smf.ols(formula="diff ~ 1", data=pd.DataFrame({"diff": diff})).fit()
    coef_diff = model.params['Intercept']
    ci_lower = model.conf_int().loc['Intercept'][0]
    ci_upper = model.conf_int().loc['Intercept'][1]
    p_value = model.pvalues['Intercept']
    cohens_d = pg.compute_effsize(avg_rocauc, pooled_rocauc, paired=True, eftype='cohen')

    print(f"Difference in pooled {metric} (statsmodels): coef={coef_diff:.4f} 95%CI[{ci_lower:.4f}, {ci_upper:.4f}], p={p_value:.4f}, Cohen's d={cohens_d:.4f}")

    stats = {"average_mean": avg_mean,
             "pooled_mean": pooled_mean,
             "diff": coef_diff,
             "95_ci_lower": ci_lower,
             "95_ci_upper": ci_upper,
             "p_value": p_value,
             "cohens_d": cohens_d,
             "N": len(y),
             "positive_class_ratio": sum(y) / len(y)}

    return stats


def plot_results(ax, df, n_samples, n_features, ylabel):
    lower = df["diff"] - df["95_ci_lower"]
    upper = df["95_ci_upper"] - df["diff"]
    yerr = np.vstack([lower.to_numpy(), upper.to_numpy()])  # shape (2, N)
    ax.errorbar(
        df["subsample_percentage"],
        df["diff"],
        yerr=yerr,
        fmt='o',
        color='black',
        capsize=2,
        linewidth=.9,
        markersize=3,
    )
    ax.hlines(0, xmin=df["subsample_percentage"].min()-5, xmax=df["subsample_percentage"].max()+5, colors='gray', linestyles='dashed', linewidth=0.8)
    ax.set_title(f"Total N: {n_samples}\nFeatures: {n_features}")
    ax.set_xticks(subsample_percentages)
    ax.set_xlabel("Dataset Size\n[\% of Original]")
    ax.set_ylabel(ylabel)


def convert_dict_to_df(results_list, subsample_percentages, dataset_name):
    """Convert list of result dicts to a DataFrame."""
    df = pd.DataFrame(results_list)
    df["subsample_percentage"] = subsample_percentages
    df["dataset_name"] = dataset_name
    return df


def format_table(df):
    """Format the results DataFrame for better readability."""
    df_formatted = df.copy()
    df_formatted["$\mu_{avg}$"] = df_formatted["average_mean"].map("{:.4f}".format)
    df_formatted["$\mu_{pooled}$"] = df_formatted["pooled_mean"].map("{:.4f}".format)
    df_formatted["$\mu_{avg}$ - $\mu_{pooled}$ [95\% CI]"] = df_formatted.apply(
        lambda row: f"{row['diff']:.4f} [{row['95_ci_lower']:.4f}, {row['95_ci_upper']:.4f}]",
        axis=1,
    )
    df_formatted["95_ci"] = df_formatted.apply(lambda row: f"[{row['95_ci_lower']:.4f}, {row['95_ci_upper']:.4f}]", axis=1)
    df_formatted["p-value"] = df_formatted["p_value"].map("{:.4f}".format)
    df_formatted["p-value"] = [f"<0.0001" if float(p_val) < 0.0001 else p_val for p_val in df_formatted["p-value"]]

    df_formatted["cohen's d"] = df_formatted["cohens_d"].map("{:.2f}".format)
    df_formatted = df_formatted.drop(columns=["95_ci_lower", "95_ci_upper"])

    df_formatted = df_formatted.rename(columns={
        "dataset_name": "Dataset",
        "subsample_percentage": "Subsample Percentage",
        "positive_class_ratio": "$p$",})

    # dataset_name, subsample_percentage, N, positive_class_ratio, average_mean, pooled_mean, diff, 95_ci_lower, 95_ci_upper, p_value, cohens_d
    df_formatted = df_formatted[["Dataset", "Subsample Percentage", "N", "$p$", "$\mu_{avg}$", "$\mu_{pooled}$", "$\mu_{avg}$ - $\mu_{pooled}$ [95\% CI]", "p-value", "cohen's d"]]
    df_formatted = df_formatted.round(4)
    
    # round to 2 decimals for: Subsample Percentage, p, "cohen's d"
    df_formatted["Subsample Percentage"] = df_formatted["Subsample Percentage"].round(2)
    df_formatted["$p$"] = df_formatted["$p$"].round(2)
    
    return df_formatted

if __name__ == "__main__":
    ylabels = {
        "rocauc": r"$AUROC_{avg}$ - $AUROC_{pooled}$",
        "prcauc": r"$AUPRC_{avg}$ - $AUPRC_{pooled}$",
        }
    for metric in ["rocauc", "prcauc"]:
        print(f"Running experiments for metric: {metric}")
        fig, axes = setup_plotting()

        subsample_percentages = [100, 90, 80, 70, 60, 50]
        X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
        X_heart_failure, y_heart_failure = load_heart_failure_data()
        X_obesity, y_obesity = load_obesity_data()
        X_heart_disease, y_heart_disease = load_heart_disease_data()
        X_cognitive_impairment, y_cognitive_impairment = load_cognitive_impairment()
        X_depression_remission, y_depression_remission = load_depression_remission()

        results_cancer = []
        results_heart_failure = []
        results_obesity = []
        results_heart_disease = []
        results_cognitive_impairment = []
        results_depression_remission = []

        for percentage in subsample_percentages:
            X_cancer_subsampled, y_cancer_subsampled = subsample_dataset(X_cancer, y_cancer, percentage, seed=1)
            stats_cancer = run_experiment(X_cancer_subsampled, y_cancer_subsampled, metric=metric)
            results_cancer.append(stats_cancer)

            X_heart_failure_subsampled, y_heart_failure_subsampled = subsample_dataset(X_heart_failure, y_heart_failure, percentage, seed=1)
            stats_heart_failure = run_experiment(X_heart_failure_subsampled, y_heart_failure_subsampled, metric=metric)
            results_heart_failure.append(stats_heart_failure)

            X_obesity_subsampled, y_obesity_subsampled = subsample_dataset(X_obesity, y_obesity, percentage, seed=1)
            stats_obesity = run_experiment(X_obesity_subsampled, y_obesity_subsampled, metric=metric)
            results_obesity.append(stats_obesity)

            X_heart_disease_subsampled, y_heart_disease_subsampled = subsample_dataset(X_heart_disease, y_heart_disease, percentage, seed=1)
            stats_heart_disease = run_experiment(X_heart_disease_subsampled, y_heart_disease_subsampled, metric=metric)
            results_heart_disease.append(stats_heart_disease)

            X_cognitive_impairment_subsampled, y_cognitive_impairment_subsampled = subsample_dataset(X_cognitive_impairment, y_cognitive_impairment, percentage, seed=1)
            stats_cognitive_impairment = run_experiment(X_cognitive_impairment_subsampled, y_cognitive_impairment_subsampled, metric=metric)
            results_cognitive_impairment.append(stats_cognitive_impairment)

            X_depression_remission_subsampled, y_depression_remission_subsampled = subsample_dataset(X_depression_remission, y_depression_remission, percentage, seed=1)
            stats_depression_remission = run_experiment(X_depression_remission_subsampled, y_depression_remission_subsampled, flipped=True, metric=metric)
            results_depression_remission.append(stats_depression_remission)


        results_cancer = convert_dict_to_df(results_cancer, subsample_percentages, dataset_name="Breast Cancer")
        results_heart_failure = convert_dict_to_df(results_heart_failure, subsample_percentages, dataset_name="Heart Failure")
        results_heart_disease = convert_dict_to_df(results_heart_disease, subsample_percentages, dataset_name="Heart Disease")
        results_obesity = convert_dict_to_df(results_obesity, subsample_percentages, dataset_name="Obesity")
        results_cognitive_impairment = convert_dict_to_df(results_cognitive_impairment, subsample_percentages, dataset_name="Cognitive Impairment")
        results_depression_remission = convert_dict_to_df(results_depression_remission, subsample_percentages, dataset_name="Depression Remission")

        plot_results(axes[0], results_cancer, X_cancer.shape[0], X_cancer.shape[1], ylabel=ylabels[metric])
        plot_results(axes[1], results_heart_failure, X_heart_failure.shape[0], X_heart_failure.shape[1], ylabel=ylabels[metric])
        plot_results(axes[2], results_heart_disease, X_heart_disease.shape[0], X_heart_disease.shape[1], ylabel=ylabels[metric])
        plot_results(axes[3], results_cognitive_impairment, X_cognitive_impairment.shape[0], X_cognitive_impairment.shape[1], ylabel=ylabels[metric])
        fig.tight_layout()
        fig.savefig(f"./results/example_subsampled_multiple_{metric}.png")
        fig.savefig(f"./results/example_subsampled_multiple_{metric}.pdf")

        df_total = pd.concat([results_cancer, results_heart_failure, results_heart_disease, results_obesity, results_cognitive_impairment, results_depression_remission], ignore_index=True)
        df_total_formatted = format_table(df_total)
        print(df_total_formatted)
        df_total_formatted.to_csv(f"./results/example_subsampled_multiple_{metric}_stats.csv", sep=",", index=False)