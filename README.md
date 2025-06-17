# UnbalancedCV
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![CI](https://github.com/melanieganz/UnbalancedCV/workflows/CI/badge.svg)](https://github.com/melanieganz/UnbalancedCV/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/melanieganz/UnbalancedCV/branch/main/graph/badge.svg)](https://codecov.io/gh/melanieganz/UnbalancedCV)
<br>

This repository is a playground for testing the effect of summarizing cross-validation results in the unbalanced data setting. 

## setting up the jupyter notebook - Franzi's setup:
```
conda create -n unbalancedcv  python=3.10
conda activate unbalancedcv
conda install jupyter
conda install pytorch
pip install matplotlib
```
now run the jupyter notebook from inside the repo dir
```
jupyter notebook
```

## Intro
This is a playground to test out the statistical properties of different ways of summarizing cross-validation results in the unbalanced data setting. The premise is to compare the summarizing metrics - accuracy, sensitivity, specificity, etc. - as well as average metrics - AUC, AUPRC - in the case where predictions are summarizes in scores per test fold vs. across all test folds.

## Problem Definition

We have a dataset D = {x, y}_{n=1}^N, where  is x_n is in R^D and y_n is in {c_0, c_1} - a binary classification vector. D is unbalanced, so it contains many more x_n with label c_0 compared to label c_1 (or vice versa). 

The question we pose is: How do we correctly do k-fold cross validation on such datasets? 

### Approach 1 : K-fold cross validation with k test sets and average the k accuracies from each test set

Let's say we want to do k-fold cross-validation. 

1. We split D into k non-overlapping data sets.
2. The k sets are then divided further into D_train (k-1 of the sets), and D_test (the kth set).
3. We train a classifier on D_train.
4. We evaluate the trained classifer on D_test and record acc_k - which denotes the accuracy of the kth fold.
5. We go back to step 1 and repeat k-1 more times and end therefor up with k estimates of the accuracy, each on a different D_test.

After this finishes we have k accuracies acc_k. We typically report the final acc_cv = mean([acc_1, acc_2, ..., acc_k])

Advantages of this approach:
* We can not only calculate a mean but also a standard deviation sof accuracies across the 10 folds. The properties of this standard deviation are though not clear.

Drawbacks of this approach: 
* In unbalanced datasets this can lead to the k folds not being balanced in the same way, even if we try to sample the k-folds in a stratified manner (see e.g. [here](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.StratifiedKFold.html))
* In can also lead to some test sets having very "easy" and clean classification points in it whereas others are hard.
   The estimation of the acc_k is based on only n/k samples instead of n, yielding a worse statistical estimate of acc_k.

### Approach 2 : K-fold cross validation with k test sets and calculate only a single accuracy across the predictiosn of all k test sets

Let's say we want to do k-fold cross-validation. 

1. We split D into k non-overlapping data sets. [*Franzi question: do we only this once at the beginning? or do we repeat this in each cycle? if we re-do this how is this different from above?* Mel answer: This is eaxctly not different from the above.]
2. The k sets are then divided further into D_train (k-1 of the sets), and D_test (the kth set).
3. We train a classifier on D_train.
4. We evaluate the trained classifer on D_test and record only the predictions \hat{y}_k on the kth set that was not used for training [*Franzi: this is the same as a above right? now whether you store the predictions or the acc_k - I don't think matters?* Mel answer: Actually, this is the whole point. Think about accuracy or the area under the ROC curve as a non-linear transformation. It matters if one does that non-linear transformation once on the whole dataset or does it several times on smaller datastes and averages. This is the pet peeve I was exctly talking about.]  
5. We go back to step 1 and repeat k-1 more times and end therefor up with k prediction vectors, each on a different D_test. [*Franzi are we going back to step 1 or step 2?* Mel answer: sorry for not being clear, step 2]

After this finishes we have n prediction estimates across the k folds. We report the final acc_cv = acc([pred_1, pred_2, ..., pred_k]). 

Advantages of this approach:
* The estimation of the acc_cv is based on all n samples.
* Especially in the unbalanced case, the imbalance is not changed.

Drawbacks of this approach: 
* We only get a single accuracy from running one k-fold cross-validation. This can though be mitigated by runnign e.g. 50 times randomized k-fold cross-validation as advertised by paper 2 below.

## Randomzied cross-validation

The two approaches above can be repeated for L times randomized k-fold cross-validation as advertised by paper 2 below. This yields then either L times K (approach 1) or L (approach 2) different accuracies over which a mean and standard deviation can be calculated.

## Related work

Some papers to look at 

"Cross-validation: what does it estimate and how well does it do it?", Bates, Hastie and Tibshirani 2022 - [arxiv](https://arxiv.org/pdf/2104.00673)

"Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines" [link](https://www.sciencedirect.com/science/article/abs/pii/S105381191630595X?casa_token=MZ9ERMMPX-oAAAAA:Qe-o-9LdL3uLNcK90To0nChJ85KEzJX9gvCnFygK4kh5h4ETdXoHXNp-i_WfM44VoAWNK_IEyvzn)

## Methods
We will simulate different cases of unbalanced data in a very simple classification setting, e.g. just usign a single feature and then adding more complicated settings.



