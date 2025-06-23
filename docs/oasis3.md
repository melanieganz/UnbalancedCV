# Investigating Cross-Validation AUC Metrics with Real Datasets - NP1
depression remission data


- [Introduction](#introduction)
  - [Setup](#setup)
  - [Load and prepare the dataset](#load-and-prepare-the-dataset)
  - [Run cross-validation](#run-cross-validation)

# Introduction

In this example, we look at the Neuropharm 1 data colletced at the NRU.
We have cortical features such as thickness, area, and curvature for 79
subjects with depression. Some of them only had one episode of
depression, others are relaåpsing. The classification task is to predict
remittence.

## Setup

<details class="code-fold">
<summary>Code</summary>

``` python
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.metrics import create_metrics
from src.cv import run_cv
```

</details>

## Load and prepare the dataset

<details class="code-fold">
<summary>Code</summary>

``` python
# Load the dataset
data_path = Path("../data/oasis3_fs_mci.tsv")
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
    f"Class distribution:\n{sum(y_binary)} positive, {len(y_binary) - sum(y_binary)} negative, ratio: {sum(y_binary) / len(y_binary):.2f}")
```

</details>

    Dataset shape: (2832, 104)
    Shape of baseline data: (1029, 104)
    Class distribution:
    755 positive, 274 negative, ratio: 0.73

## Run cross-validation

<details class="code-fold">
<summary>Code</summary>

``` python
model = LogisticRegression(max_iter=1000000)
metrics = create_metrics(["accuracy", "auc"])
results = run_cv(model, X_scaled, y_binary, metrics, n_splits=5, stratified=True, random_state=1)
results_df = pd.DataFrame(results)
print(results_df)
```

</details>

               average    pooled
    accuracy  0.820214  0.820214
    auc       0.822567  0.822845
