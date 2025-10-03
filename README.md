# UnbalancedCV
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

<br>

This repository is a playground for testing the effect of summarizing cross-validation results in the unbalanced data setting. 

## Intro
This is a playground to test out the statistical properties of different ways of summarizing performance metrics when using cross-validation. The premise is to compare the summarizing performance metrics - accuracy, sensitivity, specificity, etc. as well as AUCROC, AUPRC - in the case where predictions are summarized in performance metrics per single test fold vs. across all test folds.

## Problem Definition

We have a dataset $D = {x, y}_{i=1}^N$,  where  is $x_i$ is in $R^D$ and $y_i$ is in ${c_0, c_1}$ - a binary classification vector. $D$ is unbalanced, so it contains many more $x_i$ with label $c_0$ compared to label $c_1$ (or vice versa). 

The question we pose is: How do we correctly calculate performance metrics that summatrize the results of the k-fold cross validation on such datasets? 

### Approach 1 : $k$-fold cross validation with k test sets and average the k performance metrics from each test set

Let's say we want to do $k$-fold cross-validation. 

1. We split $D$ into $k$ non-overlapping data sets.
2. The $k$ sets are then divided further into $D_{train}$ ($k$-1 of the sets), and $D_{test}$ (the $k$ th set).
3. We train a classifier on $D_{train}$.
4. We evaluate the trained classifer on $D_{test}$ and record the performance metric $perf_k$ - which denotes the performance metric of the $k$ th fold.
5. We go back to step 1 and repeat $k$-1 more times and end therefor up with $k$ estimates of the performance metric, each on a different $D_{test}$.

After this finishes we have $k$ performance metrics $perf_k$. We typically report the final $perf_cv = mean([perf_1, perf_2, ..., perf_k])$.

Advantages of this approach:
* We can not only calculate a mean, but also a standard deviations of performance metrics across the $k$ folds. The properties of the standard deviation are though not clear.

Drawbacks of this approach: 
* In unbalanced datasets this can lead to the $k$ folds not being balanced in the same way, even if we try to sample the $k$-folds in a stratified manner (see e.g. [here](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.StratifiedKFold.html))
* It can also lead to some test sets having very "easy" and clean classification points in it whereas others are hard.
   The estimation of the performance metric $perf_k$ is based on only $n/k$ samples instead of $n$, yielding a worse statistical estimate of $perf_k$.

### Approach 2 : $k$-fold cross validation with $k$ test sets and calculate only a single performance metric across the predictions of all $k$ test sets

Let's say we want to do $k$-fold cross-validation. 

1. Just like above, we split $D$ into $k$ non-overlapping data sets. 
2. The $k$ sets are then divided further into $D_{train}$ ($k$-1 of the sets), and $D_{test}$ (the $k$ th set).
3. We train a classifier on $D_{train}$.
4. We evaluate the trained classifer on $D_{test}$ and record only the predictions $\hat{y}_k$ on the $k$ th set that was not used for training Until now we have not done anything different than in approach 1. We deviate now.  
5. We go back to step 2 and repeat it $k$-1 more times and end therefor up with $k$ predictions, each on a different $D_{test}$. 
6. After this finishes we have $n$ prediction estimates across the $k$ folds. We report the final performance metric $perf_{cv} = perf([pred_1, pred_2, ..., pred_k])$. 

Advantages of this approach:
* The estimation of the $perf_{cv}$ is based on all n samples.
* Especially in the unbalanced case, the imbalance is not changed.

Drawbacks of this approach: 
* We only get a single perfomance metric from running one k-fold cross-validation. This can though be mitigated by running e.g. 50 times randomized $k$-fold cross-validation as advertised by paper 2 below.

## Randomzied cross-validation

The two approaches above can be repeated for L times randomized k-fold cross-validation as advertised by paper 2 below. This yields then either L times K (approach 1) or L (approach 2) different accuracies over which a mean and standard deviation can be calculated.

# Theoretical considerations for Cross-Validation

## Generalization Error

With any kind of metric we aim to ultimatly estimate how the model would
perform in general. Assuming the datagenerating distribution
$p(\mathbf{x}, \mathbf{y})$, and infinite data, we can estimate the
generalization error for a trained model $f_\theta$ with the loss
$\ell(\mathbf{y}, f_\theta(\mathbf{x}))$ as

```math
\begin{align}
\mathcal{E}^{gen} = \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim p(\mathbf{x}, \mathbf{y})}[\ell(\mathbf{y}, f_\theta(\mathbf{x}))] = \int \ell(\mathbf{y}, f_\theta(\mathbf{x})) p(\mathbf{x}, \mathbf{y}) d\mathbf{x}d\mathbf{y}
\end{align}
```

The problem here is that we do not know the underlying distribution. An
alternative would be to approximate the generalization error. This can
be either done on a large enough hold-out set, or using k-fold
cross-validation.

## K‑Fold Cross‑Validation

In K-fold cross-validation (CV), we split our dataset $\mathcal{D}$ into
$k$ folds and use $k$-1 folds for training, and one for testing. This is
repeated $k$  times so that each fold was used for testing once. 
There are two strategies that can be used to compute the generalization
error matching our approach 1 and 2 described in the ReadMe. One is to evaulate the loss for each of the $k$
test-folds $\mathcal{D}^{test}_k$ and then average them. We refer to this as
$\mathcal{E}^{gen}_{average}$. The other option is to pool all predictions from the hold-out sets and evaluate the loss on the the pooled data, we refer to this as $\mathcal{E}^{gen}_{pooled}$. 

### Aproach 1: Averaged Generalization Error

The generalization error can then be approximated as the weighted
average of the test errors in each hold-out set $\mathcal{D}^{test}_k$:

```math
\begin{align}
\mathcal{E}^{gen} \approx \mathcal{E}^{gen}_{average} = \sum_{k=1}^{K} \frac{N_k}{N}\mathcal{E}^{test}_k
\end{align}
```

with $N_k$ being the number of test samples per fold, and $N$ the
total number of samples, and 

```math
\begin{align}
\mathcal{E}^{test} = \frac{1}{N_k} \sum_{j=1}^{N_k} \ell(y_{k,j}, f_\theta(x_{k,j})), 
\end{align}
```

where $j$ indexes into the $k$-th fold.

### Approach 2: Pooled Generalization Error

The generalization error can also be approximated by the loss evaluate don the whole dataset, which is pooled from all test sets:

```math
\begin{align}
\mathcal{E}^{gen} \approx \mathcal{E}^{gen}_{pooled} = \frac{1}{N} \sum_{i=1}^{N} \ell(y_{i}, f_\theta(x_{i})), 
\end{align}
```

where $i$ indexes all $N$ samples.


## Theoretical Example

Let us define different scenarios for the two approaches.First we will choose our loss to be accuracy, which is defined as the fraction of correct predictions. At the single sample level, the loss is hence defined as:

```math
\begin{align}
\ell(y, f_\theta(x)) = \begin{cases}
1 & \text{if } y = f_\theta(x) \\
0 & \text{otherwise}  
\end{cases} = \mathbb{1}(y, f_\theta(x)),
\end{align}
```

where $\mathbb{1}(y, f_\theta(x))$ is the indicator function that returns 1 if the prediction is correct and 0 otherwise.  

The averaged generalization error for this loss is then:

```math
\begin{align}
\mathcal{E}^{gen}_{average} &= 
\sum_{k=1}^{K} \frac{N_k}{N} \mathcal{E}^{test}_k =
\sum_{k=1}^{K} \frac{N_k}{N} \frac{1}{N_k} \sum_{j=1}^{N_k} \ell(y_{k,j}, f_\theta(x_{k,j})) \\
&=\frac{1}{N} \sum_{k=1}^{K} \sum_{j=1}^{N_k} \ell\mathbb{1}(y_{k,j}, f_\theta(x_{k,j})) \\
&=\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_{i}, f_\theta(x_{i})).
\end{align}
```

The pooled generalization error is then:

```math
\mathcal{E}^{gen}_{pooled} = \frac{1}{N} \sum_{i=1}^{N} \ell(y_{i}, f_\theta(x_{i})) =
\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_{i}, f_\theta(x_{i})).
```

In this case, the averaged generalization error is equal to the pooled generalization error, since we are evaluating the idicator funtion for each individual sample and then summing over all samples in the end.

Another example is the case of a loss that is not an indicator function, such as the squared error loss $\ell(y, f_\theta(x)) = (y - f_\theta(x))^2$. In this case, the averaged generalization error and the pooled generalization error will again yield the same results. In general, for any loss that is a function of the individual predictions, the two approaches will yield the same results, since they both evaluate the loss on each individual sample and then average over all samples.

Now in the case of a loss that is not only dependent on the individual samples, the two approaches can yield different results. For example, if we use the area under the receiver operating characteristic curve (AUCROC) as our loss. This loss is defined as the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample, $\ell(y, f_\theta(x)) = \mathbb{P}(f_\theta(x_{j}) > f_\theta(x_{i})|y_i=0, y_j=1)$, where $y_{i}$ and $y_{j}$ are negative and positive samples, respectively.

Across a given sample, the expected AUCROC can be defined as:        

```math
\begin{align}
\ell(y, f_\theta(x)) &= 
\frac{1}{N_{neg} N_{pos}} \sum_{i=1}^{N} \sum_{j=1}^{N} \mathbb{1}(f_\theta(x_{j}) > f_\theta(x_{i})) \mathbb{1}(y_i=0, y_j=1) \\
&= \frac{1}{N_{neg} N_{pos}} \sum_{i=1}^{N_{neg}} \sum_{j=1}^{N_{pos}} \mathbb{1}(f_\theta(x_{j}) > 
f_\theta(x_{i})),
\end{align}
```

where $N_{pos}$ and $N_{neg}$ are the number of positive and negative samples, respectively (see [Le Dell et al., 2015](#references)).

## Code Examples 

We will simulate different cases of unbalanced data in a very simple classification setting, e.g. just using a single feature and then adding more complicated settings.

In the notebooks and scripts we show an example of how the two approaches differ in practice. 

### Environment setup:
```bash
# create the environment with the right python version
conda create -n unbalancedcv python=3.10

# Activate the environment
conda activate unbalancedcv

# Install from requirements.txt using conda
conda install --file requirements.txt

# install the local package
pip install -e .
```

### Reproducing the results

To reproduce the simulations, run the script `notebooks/simulation_theoretical_difference.py` from the root directory.
```bash
# From the root directory run:
python notebooks/simulation_theoretical_difference.py
```
To the real world examples, run the script `notebooks/examples.py`. However, you will need the files with the data which will obrait from the specific datasets.
```bash
python notebooks/examples.py
```
Both scripts will place the figures in the root directory.

## References
- Le Dell, E., Petersen, M., & van der Laan, M.(2015). "Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates." *Electronic Journal of Statistics*, 9(1), 1583–1607. [https://doi.org/10.1214/15-EJS1035](https://doi.org/10.1214/15-EJS1035)

- "Cross-validation: what does it estimate and how well does it do it?", Bates, Hastie and Tibshirani 2022 - [arxiv](https://arxiv.org/pdf/2104.00673)

- "Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines" [link](https://www.sciencedirect.com/science/article/abs/pii/S105381191630595X?casa_token=MZ9ERMMPX-oAAAAA:Qe-o-9LdL3uLNcK90To0nChJ85KEzJX9gvCnFygK4kh5h4ETdXoHXNp-i_WfM44VoAWNK_IEyvzn)
