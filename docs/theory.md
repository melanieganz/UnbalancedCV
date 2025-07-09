# Theoretical considerations for Cross-Validation


# Generalization Error

With any kind of metric we aim to ultimatly estimate how the model would
perform in general. Assuming the datagenerating distribution
$p(\mathbf{x}, \mathbf{y})$, and infinite data, we can estimate the
generalization error for a trained model $f_\theta$ with the loss
$\ell(\mathbf{y}, f_\theta(\mathbf{x}))$ as

$$
\mathcal{E}^{gen} = \mathbb{E}_{(\mathbf{x},\mathbf{y}) \sim p(\mathbf{x}, \mathbf{y})}[\ell(\mathbf{y}, f_\theta(\mathbf{x}))]
= \int \ell(\mathbf{y}, f_\theta(\mathbf{x})) p(\mathbf{x}, \mathbf{y}) d\mathbf{x}d\mathbf{y}
$$

The problem here is that we do not know the underlying distirubiton. An
alternative would be to approximate the generalization error. This can
be either done on a large enough hold-out set, or using k-fold
cross-validation.

# K‑Fold Cross‑Validation

In K-fold cros-validation (CV), we split our dataset $\mathcal{D}$ into
K folds and use K-1 folds for training, and one for testing. This is
repeated K times so that each fold was used for testing once. On each of
the test-folds $\mathcal{D}^{test}_k$, we can compute the test error.
There are two strategies that can be used to compute the generalization
error matching our approach 1 and 2 describe din the ReadMe. One is to evaulate the loss for each fold and average then. We refere to this as
$\mathcal{E}^{gen}_{average}$. The other option is to pool all predictions from the hold-out sets and evaluate the Loss on the the pooled data, we refer to this as $\mathcal{E}^{gen}_{pooled}$. 

## Aproach 1: Averaged Generalization Error

The generalization error can then be approximated as the weighted
average of the test errors in each hold-out set $\mathcal{D}^{test}_k$:

$\mathcal{E}^{gen} \approx \mathcal{E}^{gen}_{average} = \sum_{k=1}^{K} \frac{N_k}{N}\mathcal{E}^{test}_k$  with $N_k$ being the number of test samples per fold, and $N$ the
total number of samples, and $
\mathcal{E}^{test} = \frac{1}{N_k} \sum_{i}^{N_k} \ell(y_{k,i}, f_\theta(x_{k,i}))$

## Approach 2: Pooled Generalization Error

The generalization error can also be approximated by the loss evaluate don the whole dataset, which is pooled from all test sets:

$\mathcal{E}^{gen} \approx \mathcal{E}^{gen}_{pooled} = \frac{1}{N} \sum_{i=1}^{N} \ell(y_{i}, f_\theta(x_{i}))$
