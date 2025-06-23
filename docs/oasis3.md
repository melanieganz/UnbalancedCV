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
print(f"Columns: {df.columns.tolist()}")

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
    Columns: ['subject', 'session', 'age', 'cognitiveyly_normal', '3rd-Ventricle', '4th-Ventricle', 'Brain-Stem', 'CC_Anterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Mid_Posterior', 'CC_Posterior', 'CSF', 'Left-Accumbens-area', 'Left-Amygdala', 'Left-Caudate', 'Left-Cerebellum-Cortex', 'Left-Cerebellum-White-Matter', 'Left-Cerebral-White-Matter', 'Left-Hippocampus', 'Left-Inf-Lat-Vent', 'Left-Lateral-Ventricle', 'Left-Pallidum', 'Left-Putamen', 'Left-Thalamus', 'Left-VentralDC', 'Left-choroid-plexus', 'Right-Accumbens-area', 'Right-Amygdala', 'Right-Caudate', 'Right-Cerebellum-Cortex', 'Right-Cerebellum-White-Matter', 'Right-Cerebral-White-Matter', 'Right-Hippocampus', 'Right-Inf-Lat-Vent', 'Right-Lateral-Ventricle', 'Right-Pallidum', 'Right-Putamen', 'Right-Thalamus', 'Right-VentralDC', 'Right-choroid-plexus', 'WM-hypointensities', 'ctx-lh-caudalanteriorcingulate', 'ctx-lh-caudalmiddlefrontal', 'ctx-lh-cuneus', 'ctx-lh-entorhinal', 'ctx-lh-fusiform', 'ctx-lh-inferiorparietal', 'ctx-lh-inferiortemporal', 'ctx-lh-insula', 'ctx-lh-isthmuscingulate', 'ctx-lh-lateraloccipital', 'ctx-lh-lateralorbitofrontal', 'ctx-lh-lingual', 'ctx-lh-medialorbitofrontal', 'ctx-lh-middletemporal', 'ctx-lh-paracentral', 'ctx-lh-parahippocampal', 'ctx-lh-parsopercularis', 'ctx-lh-parsorbitalis', 'ctx-lh-parstriangularis', 'ctx-lh-pericalcarine', 'ctx-lh-postcentral', 'ctx-lh-posteriorcingulate', 'ctx-lh-precentral', 'ctx-lh-precuneus', 'ctx-lh-rostralanteriorcingulate', 'ctx-lh-rostralmiddlefrontal', 'ctx-lh-superiorfrontal', 'ctx-lh-superiorparietal', 'ctx-lh-superiortemporal', 'ctx-lh-supramarginal', 'ctx-lh-transversetemporal', 'ctx-rh-caudalanteriorcingulate', 'ctx-rh-caudalmiddlefrontal', 'ctx-rh-cuneus', 'ctx-rh-entorhinal', 'ctx-rh-fusiform', 'ctx-rh-inferiorparietal', 'ctx-rh-inferiortemporal', 'ctx-rh-insula', 'ctx-rh-isthmuscingulate', 'ctx-rh-lateraloccipital', 'ctx-rh-lateralorbitofrontal', 'ctx-rh-lingual', 'ctx-rh-medialorbitofrontal', 'ctx-rh-middletemporal', 'ctx-rh-paracentral', 'ctx-rh-parahippocampal', 'ctx-rh-parsopercularis', 'ctx-rh-parsorbitalis', 'ctx-rh-parstriangularis', 'ctx-rh-pericalcarine', 'ctx-rh-postcentral', 'ctx-rh-posteriorcingulate', 'ctx-rh-precentral', 'ctx-rh-precuneus', 'ctx-rh-rostralanteriorcingulate', 'ctx-rh-rostralmiddlefrontal', 'ctx-rh-superiorfrontal', 'ctx-rh-superiorparietal', 'ctx-rh-superiortemporal', 'ctx-rh-supramarginal', 'ctx-rh-transversetemporal']
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
