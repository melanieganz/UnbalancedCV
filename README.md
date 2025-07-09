# UnbalancedCV
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![CI](https://github.com/melanieganz/UnbalancedCV/workflows/CI/badge.svg)](https://github.com/melanieganz/UnbalancedCV/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/melanieganz/UnbalancedCV/branch/main/graph/badge.svg)](https://codecov.io/gh/melanieganz/UnbalancedCV)
<br>
Authors: Ruben Doerfel, Franziska Meier and Melanie Ganz

This repository is a playground for testing the effect of summarizing cross-validation results in the unbalanced data setting. 

## Intro
This is a playground to test out the statistical properties of different ways of summarizing performance metrics when using cross-validation. The premise is to compare the summarizing performance metrics - accuracy, sensitivity, specificity, etc. as well as AUCROC, AUPRC - in the case where predictions are summarized in performance metrics per single test fold vs. across all test folds.

## Problem Definition

We have a dataset $D = {x, y}_{i=1}^N$,  where  is $x_i$ is in $R^D$ and $y_i$ is in ${c_0, c_1}$ - a binary classification vector. $D$ is unbalanced, so it contains many more $x_i$ with label $c_0$ compared to label $c_1$ (or vice versa). 

The question we pose is: How do we correctly calculate performance metrics that summatrize the results of the k-fold cross validation on such datasets? 

### Approach 1 : $k$-fold cross validation with k test sets and average the k performance metrics from each test set

Let's say we want to do $k$-fold cross-validation. 

1. We split $D$ into $k$ non-overlapping data sets.
2. The $k$ sets are then divided further into $D_{train}$ ($k$-1 of the sets), and $D_test$ (the $k$ th set).
3. We train a classifier on $D_{train}$.
4. We evaluate the trained classifer on $D_{test}$ and record the performance metric $perf_k$ - which denotes the performance metric of the $k$ th fold.
5. We go back to step 1 and repeat $k$-1 more times and end therefor up with $k$ estimates of the performance metric, each on a different $D_{test}$.

After this finishes we have $k$ performance metrics $perf_k$. We typically report the final $perf_cv = mean([perf_1, perf_2, ..., perf_k])$.

Advantages of this approach:
* We can not only calculate a mean, but also a standard deviations of performance metrics across the 10 folds. The properties of the standard deviation are though not clear.

Drawbacks of this approach: 
* In unbalanced datasets this can lead to the $k$ folds not being balanced in the same way, even if we try to sample the $k$-folds in a stratified manner (see e.g. [here](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.StratifiedKFold.html))
* In can also lead to some test sets having very "easy" and clean classification points in it whereas others are hard.
   The estimation of the performance metric $perf_k$ is based on only $n/k$ samples instead of $n$, yielding a worse statistical estimate of $perf_k$.

### Approach 2 : $k$-fold cross validation with $k$ test sets and calculate only a single performance metric across the predictions of all $k$ test sets

Let's say we want to do $k$-fold cross-validation. 

1. Just like above, we split $D$ into $k$ non-overlapping data sets. 
2. The $k$ sets are then divided further into $D_{train}$ ($k$-1 of the sets), and $D_{test}$ (the $k$th set).
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

## Related work

Some papers to look at 

"Cross-validation: what does it estimate and how well does it do it?", Bates, Hastie and Tibshirani 2022 - [arxiv](https://arxiv.org/pdf/2104.00673)

"Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines" [link](https://www.sciencedirect.com/science/article/abs/pii/S105381191630595X?casa_token=MZ9ERMMPX-oAAAAA:Qe-o-9LdL3uLNcK90To0nChJ85KEzJX9gvCnFygK4kh5h4ETdXoHXNp-i_WfM44VoAWNK_IEyvzn)

## Methods
We will simulate different cases of unbalanced data in a very simple classification setting, e.g. just usign a single feature and then adding more complicated settings.

## Environment setup:
```bash
# create the environment with the right python version
conda create -n unbalancedcv python=3.10

# Activate the environment
conda activate unbalancedcv

# Install from requirements.txt using conda
conda install --file requirements.txt
```
now run the jupyter notebook from inside the repo dir

```
jupyter notebook
```
