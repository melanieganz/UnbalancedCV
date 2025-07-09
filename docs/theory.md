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

The problem here is that we do not know the underlying distribution. An
alternative would be to approximate the generalization error. This can
be either done on a large enough hold-out set, or using k-fold
cross-validation.

# K‑Fold Cross‑Validation

In K-fold cross-validation (CV), we split our dataset $\mathcal{D}$ into
$k$ folds and use $k$-1 folds for training, and one for testing. This is
repeated $k$  times so that each fold was used for testing once. 
There are two strategies that can be used to compute the generalization
error matching our approach 1 and 2 described in the ReadMe. One is to evaulate the loss for each of the $k$
test-folds $\mathcal{D}^{test}_k$ and then average them. We refer to this as
$\mathcal{E}^{gen}_{average}$. The other option is to pool all predictions from the hold-out sets and evaluate the loss on the the pooled data, we refer to this as $\mathcal{E}^{gen}_{pooled}$. 

## Aproach 1: Averaged Generalization Error

The generalization error can then be approximated as the weighted
average of the test errors in each hold-out set $\mathcal{D}^{test}_k$:

$\mathcal{E}^{gen} \approx \mathcal{E}^{gen}_{average} = \sum_{k=1}^{K} \frac{N_k}{N}\mathcal{E}^{test}_k$  with $N_k$ being the number of test samples per fold, and $N$ the
total number of samples, and $
\mathcal{E}^{test} = \frac{1}{N_k} \sum_{j=1}^{N_k} \ell(y_{k,j}, f_\theta(x_{k,j}))$, where $j$ indexes into the $k$-th fold.

## Approach 2: Pooled Generalization Error

The generalization error can also be approximated by the loss evaluate don the whole dataset, which is pooled from all test sets:

$\mathcal{E}^{gen} \approx \mathcal{E}^{gen}_{pooled} = \frac{1}{N} \sum_{i=1}^{N} \ell(y_{i}, f_\theta(x_{i}))$, where $i$ indexes all $N$ samples.


## Theoretical Example

Let us define different scenarios for the two approaches.First we will choose our loss to be accuracy, which is defined as the fraction of correct predictions. At the single sample level, the loss is hence defined as:

$\ell(y, f_\theta(x)) = \begin{cases}
1 & \text{if } y = f_\theta(x) \\
0 & \text{otherwise}  
\end{cases} = \mathbb{1}(y, f_\theta(x))$,

where $\mathbb{1}(y, f_\theta(x))$ is the indicator function that returns 1 if the prediction is correct and 0 otherwise.  

The averaged generalization error for this loss is then:

$$\mathcal{E}^{gen}_{average} = 
\sum_{k=1}^{K} \frac{N_k}{N} \mathcal{E}^{test}_k =
 \sum_{k=1}^{K} \frac{N_k}{N} \frac{1}{N_k} \sum_{j=1}^{N_k} \ell(y_{k,j}, f_\theta(x_{k,j})) = 
 \frac{1}{N} \sum_{k=1}^{K} \sum_{j=1}^{N_k} \ell\mathbb{1}(y_{k,j}, f_\theta(x_{k,j})) = 
 \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_{i}, f_\theta(x_{i}))$$.


 The pooled generalization error is then:
$$\mathcal{E}^{gen}_{pooled} = \frac{1}{N} \sum_{i=1}^{N} \ell(y_{i}, f_\theta(x_{i})) =
\frac{1}{N} \sum_{i=1}^{N} \mathbb{1}(y_{i}, f_\theta(x_{i}))$$.

In this case, the averaged generalization error is equal to the pooled generalization error, since we are evaluating the idicator funtion for each individual sample and then summing over all samples in the end.

Another example is the case of a loss that is not an indicator function, such as the squared error loss $\ell(y, f_\theta(x)) = (y - f_\theta(x))^2$. In this case, the averaged generalization error and the pooled generalization error will again yield the same results. In general, for any loss that is a function of the individual predictions, the two approaches will yield the same results, since they both evaluate the loss on each individual sample and then average over all samples.

Now in the case of a loss that is not only dependent on the individual samples, the two approaches can yield different results. For example, if we use the area under the receiver operating characteristic curve (AUCROC) as our loss. This loss is defined as the probability that a randomly chosen positive sample is ranked higher than a randomly chosen negative sample, $\ell(y, f_\theta(x)) = \mathbb{P}(f_\theta(x_{j}) > f_\theta(x_{i})|y_i=0, y_j=1)$, where $y_{i}$ and $y_{j}$ are negative and positive samples, respectively.

Across a given sample, the expected AUCROC can be defined as:        

$\ell(y, f_\theta(x)) = \frac{1}{N_{neg} N_{pos}} \sum_{i=1}^{N} \sum_{j=1}^{N} \mathbb{1}(f_\theta(x_{j}) > f_\theta(x_{i})) \mathbb{1}(y_i=0, y_j=1) = \frac{1}{N_{neg} N_{pos}} \sum_{i=1}^{N_{neg}} \sum_{j=1}^{N_{pos}} \mathbb{1}(f_\theta(x_{j}) > f_\theta(x_{i}))$ ,

where $N_{pos}$ and $N_{neg}$ are the number of positive and negative samples, respectively (see [Le Dell et al., 2015](#references)).



## Code Examples 

In the notebooks we show an example of how the two approaches differ in practice. 

## References
- Le Dell, E., Petersen, M., & van der Laan, M.(2015). "Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates." *Electronic Journal of Statistics*, 9(1), 1583–1607. [https://doi.org/10.1214/15-EJS1035](https://doi.org/10.1214/15-EJS1035)
