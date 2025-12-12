#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import statsmodels.formula.api as smf
import pingouin as pg
from src.cv import run_repeated_cv
from src import simulations
from src.metrics import create_metrics

#%% functions

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

#%% Simulate data
# set up the dataset parameters
n_samples = 500
pos_ratio = 0.1
mu0 = -1
sigma0 = 1
mu1 = 1
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

# %% repeat this with cross-validation
results = run_experiment(X, y, flipped=False, metric="rocauc")
print(results)




#%% sample a large dataset to get the "True" prediction error
results = []

for seed in range(1, 101):
    # simulate new set of data
    X, y = simulations.simulate_dataset(n_samples, pos_ratio, mu0, sigma0, mu1, sigma1, seed=seed)

    # setup model
    model = LogisticRegression()
    metrics = create_metrics(["accuracy", "rocauc", "prcauc"])

    # run cv
    result = run_cv(model, X, y, metrics, n_splits=5, stratified=True, random_state=1)

    # get "true" values from large dataset
    X_large, y_large = simulations.simulate_dataset(100000, pos_ratio, mu0, sigma0, mu1, sigma1, seed=seed)
    model.fit(X_large, y_large)
    y_pred_large = model.predict_proba(X_large)[:, 1]
    true_rocauc = metrics["rocauc"](y_large, None, y_pred_large)
    true_prcauc = metrics["prcauc"](y_large, None, y_pred_large)
    true_accuracy = metrics["accuracy"](y_large, (y_pred_large >= 0.5).astype(int), None)
    
    true_result = {"rocauc": true_rocauc,
                   "prcauc": true_prcauc,
                   "accuracy": true_accuracy}

    # combine
    df_average = pd.DataFrame([result["average"]])
    df_average["seed"] = seed
    df_average["method"] = "average"
    df_pooled = pd.DataFrame([result["pooled"]])
    df_pooled["seed"] = seed
    df_pooled["method"] = "pooled"
    df_true = pd.DataFrame([true_result])
    df_true["seed"] = seed
    df_true["method"] = "true"
    df_result = pd.concat([df_average, df_pooled, df_true], ignore_index=True)
    results.append(df_result)


results_df = pd.concat(results, ignore_index=True)

# %%
# visualize results
import seaborn as sns
import statsmodels.formula.api as smf

df_average_true = results_df[results_df["method"].isin(["average", "true"])]
df_pooled_true = results_df[results_df["method"].isin(["pooled", "true"])]

# fit an lm 

model_avg = smf.ols(formula="rocauc ~ C(method)", data=df_average_true).fit()
print(model_avg.summary())
# %%
model_pooled = smf.ols(formula="rocauc ~ C(method)", data=df_pooled_true).fit()
print(model_pooled.summary())
# %%
sns.boxplot(x="method", y="rocauc", data=results_df)
# %%
