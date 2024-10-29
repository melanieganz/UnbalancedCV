# UnbalancedCV
This repository is a playground for testing the effect of summarizing cross-validation results in the unbalanced data setting. 

## Intro
This is a playground to test out the statistical properties of different ways of summarizing cross-validation results in the unbalanced data setting. The premise is to compare the summarizing metrics - accuracy, sensitivity, specificity, etc. - as well as average metrics - AUC, AUPRC - in the case where predictions are summarizes in scores per test fold vs. across all test folds.

## Problem Definition

We have a dataset D = {x, y}_{n=1}^N, where  is x_n is in R^D and y_n is in {c_0, c_1} - a binary classification vector. D is unbalanced, so it contains many more x_n with label c_0 compared to label c_1 (or vice versa). 

The question we pose is: How do we correctly do k-fold cross validation on such datasets? 

### Approach 1 : Random splits into train and test sets and average the k accuracies from

Let's say we want to do 10-fold cross-validation. 


1. We split D into D_train, and D_test, where we determine D_test by randomly sampling 10% of the data points in D. D_train is the rest.
2. we train a classifier on D_train
3. we evaluate the trained classifer on D_test and record acc_k - which denotes the accuracy of the kth fold
4. we go back to step 1 and repeat 9 more times

After this finishes we have 10 accuracies acc_k. We typically report the final acc_cv = mean([acc_1, acc_2, ..., acc_10])

Drawbacks of this approach: ?

### Approach 2 ?


## Related work

Some papers to look at 

"Cross-validation: what does it estimate and how well does it do it?", Bates, Hastie and Tibshirani 2022 - [arxiv](https://arxiv.org/pdf/2104.00673)

"Assessing and tuning brain decoders: Cross-validation, caveats, and guidelines" [link](https://www.sciencedirect.com/science/article/abs/pii/S105381191630595X?casa_token=MZ9ERMMPX-oAAAAA:Qe-o-9LdL3uLNcK90To0nChJ85KEzJX9gvCnFygK4kh5h4ETdXoHXNp-i_WfM44VoAWNK_IEyvzn)

## Methods
We will simulate different cases of unbalanced data in a very simple classification setting, e.g. just usign a single feature and then adding more complicated settings.



