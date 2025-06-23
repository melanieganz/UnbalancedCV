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

from src.metrics import create_metrics
from src.cv import run_cv
```

</details>

## Load and prepare the dataset

<details class="code-fold">
<summary>Code</summary>

``` python
# Load the dataset
data_path = Path("../data/np1_fs_mdd_episode.csv")
df = pd.read_csv(data_path)

print(f"Dataset shape: {df.shape}")

# drop diagnosis:
df = df.drop(columns=["diagnosis"])

# convert to numpy arrays
X = df.drop(columns=["mdd_episode"]).to_numpy()
y = df["mdd_episode"].to_numpy()

# we need to convert the target to a binary classification task
y_binary = (y == "Recurrent").astype(int)
# Check class balance
print(
    f"Class distribution:\n{sum(y_binary)} positive, {len(y_binary) - sum(y_binary)} negative, ratio: {sum(y_binary) / len(y_binary):.2f}")
```

</details>

    Dataset shape: (79, 163)
    Class distribution:
    47 positive, 32 negative, ratio: 0.59

## Run cross-validation

<details class="code-fold">
<summary>Code</summary>

``` python
model = LogisticRegression(max_iter=10000)
metrics = create_metrics(["accuracy", "auc"])
results = run_cv(model, X, y_binary, metrics, n_splits=5, stratified=True, random_state=1)
results_df = pd.DataFrame(results)
print(results_df)
```

</details>

               average    pooled
    accuracy  0.531646  0.531646
    auc       0.569299  0.567154
