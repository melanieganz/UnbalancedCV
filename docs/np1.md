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
print(f"Columns: {df.columns.tolist()}")

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
    Columns: ['lh.bankssts.thickness', 'lh.caudalanteriorcingulate.thickness', 'lh.caudalmiddlefrontal.thickness', 'lh.cuneus.thickness', 'lh.entorhinal.thickness', 'lh.fusiform.thickness', 'lh.inferiorparietal.thickness', 'lh.inferiortemporal.thickness', 'lh.isthmuscingulate.thickness', 'lh.lateraloccipital.thickness', 'lh.lateralorbitofrontal.thickness', 'lh.lingual.thickness', 'lh.medialorbitofrontal.thickness', 'lh.middletemporal.thickness', 'lh.parahippocampal.thickness', 'lh.paracentral.thickness', 'lh.parsopercularis.thickness', 'lh.parsorbitalis.thickness', 'lh.parstriangularis.thickness', 'lh.pericalcarine.thickness', 'lh.postcentral.thickness', 'lh.posteriorcingulate.thickness', 'lh.precentral.thickness', 'lh.precuneus.thickness', 'lh.rostralanteriorcingulate.thickness', 'lh.rostralmiddlefrontal.thickness', 'lh.superiorfrontal.thickness', 'lh.superiorparietal.thickness', 'lh.superiortemporal.thickness', 'lh.supramarginal.thickness', 'lh.frontalpole.thickness', 'lh.temporalpole.thickness', 'lh.transversetemporal.thickness', 'lh.insula.thickness', 'lh.whole_hemisphere.thickness', 'lh.bankssts.surface_area', 'lh.caudalanteriorcingulate.surface_area', 'lh.caudalmiddlefrontal.surface_area', 'lh.cuneus.surface_area', 'lh.entorhinal.surface_area', 'lh.fusiform.surface_area', 'lh.inferiorparietal.surface_area', 'lh.inferiortemporal.surface_area', 'lh.isthmuscingulate.surface_area', 'lh.lateraloccipital.surface_area', 'lh.lateralorbitofrontal.surface_area', 'lh.lingual.surface_area', 'lh.medialorbitofrontal.surface_area', 'lh.middletemporal.surface_area', 'lh.parahippocampal.surface_area', 'lh.paracentral.surface_area', 'lh.parsopercularis.surface_area', 'lh.parsorbitalis.surface_area', 'lh.parstriangularis.surface_area', 'lh.pericalcarine.surface_area', 'lh.postcentral.surface_area', 'lh.posteriorcingulate.surface_area', 'lh.precentral.surface_area', 'lh.precuneus.surface_area', 'lh.rostralanteriorcingulate.surface_area', 'lh.rostralmiddlefrontal.surface_area', 'lh.superiorfrontal.surface_area', 'lh.superiorparietal.surface_area', 'lh.superiortemporal.surface_area', 'lh.supramarginal.surface_area', 'lh.frontalpole.surface_area', 'lh.temporalpole.surface_area', 'lh.transversetemporal.surface_area', 'lh.insula.surface_area', 'lh.whole_hemisphere.surface_area', 'rh.bankssts.thickness', 'rh.caudalanteriorcingulate.thickness', 'rh.caudalmiddlefrontal.thickness', 'rh.cuneus.thickness', 'rh.entorhinal.thickness', 'rh.fusiform.thickness', 'rh.inferiorparietal.thickness', 'rh.inferiortemporal.thickness', 'rh.isthmuscingulate.thickness', 'rh.lateraloccipital.thickness', 'rh.lateralorbitofrontal.thickness', 'rh.lingual.thickness', 'rh.medialorbitofrontal.thickness', 'rh.middletemporal.thickness', 'rh.parahippocampal.thickness', 'rh.paracentral.thickness', 'rh.parsopercularis.thickness', 'rh.parsorbitalis.thickness', 'rh.parstriangularis.thickness', 'rh.pericalcarine.thickness', 'rh.postcentral.thickness', 'rh.posteriorcingulate.thickness', 'rh.precentral.thickness', 'rh.precuneus.thickness', 'rh.rostralanteriorcingulate.thickness', 'rh.rostralmiddlefrontal.thickness', 'rh.superiorfrontal.thickness', 'rh.superiorparietal.thickness', 'rh.superiortemporal.thickness', 'rh.supramarginal.thickness', 'rh.frontalpole.thickness', 'rh.temporalpole.thickness', 'rh.transversetemporal.thickness', 'rh.insula.thickness', 'rh.whole_hemisphere.thickness', 'rh.bankssts.surface_area', 'rh.caudalanteriorcingulate.surface_area', 'rh.caudalmiddlefrontal.surface_area', 'rh.cuneus.surface_area', 'rh.entorhinal.surface_area', 'rh.fusiform.surface_area', 'rh.inferiorparietal.surface_area', 'rh.inferiortemporal.surface_area', 'rh.isthmuscingulate.surface_area', 'rh.lateraloccipital.surface_area', 'rh.lateralorbitofrontal.surface_area', 'rh.lingual.surface_area', 'rh.medialorbitofrontal.surface_area', 'rh.middletemporal.surface_area', 'rh.parahippocampal.surface_area', 'rh.paracentral.surface_area', 'rh.parsopercularis.surface_area', 'rh.parsorbitalis.surface_area', 'rh.parstriangularis.surface_area', 'rh.pericalcarine.surface_area', 'rh.postcentral.surface_area', 'rh.posteriorcingulate.surface_area', 'rh.precentral.surface_area', 'rh.precuneus.surface_area', 'rh.rostralanteriorcingulate.surface_area', 'rh.rostralmiddlefrontal.surface_area', 'rh.superiorfrontal.surface_area', 'rh.superiorparietal.surface_area', 'rh.superiortemporal.surface_area', 'rh.supramarginal.surface_area', 'rh.frontalpole.surface_area', 'rh.temporalpole.surface_area', 'rh.transversetemporal.surface_area', 'rh.insula.surface_area', 'rh.whole_hemisphere.surface_area', 'lh.lateral-ventricle.volume', 'lh.cerebellum-cortex.volume', 'lh.thalamus-proper.volume', 'lh.caudate.volume', 'lh.putamen.volume', 'lh.pallidum.volume', 'lh.hippocampus.volume', 'lh.amygdala.volume', 'lh.accumbens-area.volume', 'rh.lateral-ventricle.volume', 'rh.cerebellum-cortex.volume', 'rh.thalamus-proper.volume', 'rh.caudate.volume', 'rh.putamen.volume', 'rh.pallidum.volume', 'rh.hippocampus.volume', 'rh.amygdala.volume', 'rh.accumbens-area.volume', 'icv.volume', 'diagnosis', 'age', 'hamd6', 'mdd_episode']
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
