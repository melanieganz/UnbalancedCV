# Investigating Cross-Validation AUC Metrics with Real Datasets


- [Introduction](#introduction)
  - [Setup](#setup)
  - [Load and prepare the dataset](#load-and-prepare-the-dataset)
  - [Run cross-validation](#run-cross-validation)

# Introduction

In this example, we look at the Breast Cancer dataset from scikit-learn.
The classification task is to predict whether the subject has cancer or
not.

## Setup

<details class="code-fold">
<summary>Code</summary>

``` python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Import our custom functions
from src.metrics import create_metrics
from src.cv import run_cv
```

</details>

## Load and prepare the dataset

<details class="code-fold">
<summary>Code</summary>

``` python
X, y = load_breast_cancer(return_X_y=True)

# print dataset shape
print(f"Dataset shape: {X.shape}")
# Check class balance
print(f"Class distribution:\n{sum(y)} positive, {len(y) - sum(y)} negative, ratio: {sum(y) / len(y):.2f}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

</details>

    Dataset shape: (569, 30)
    Class distribution:
    357 positive, 212 negative, ratio: 0.63

## Run cross-validation

<details class="code-fold">
<summary>Code</summary>

``` python
model = LogisticRegression()
metrics = create_metrics(["accuracy", "auc"])
results = run_cv(model, X_scaled, y, metrics, n_splits=5, stratified=True, random_state=1)

results_df = pd.DataFrame(results)
print(results_df)
```

</details>

               average    pooled
    accuracy  0.978910  0.978910
    auc       0.995428  0.995098
