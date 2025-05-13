# Lecture 1
## Introduction
### Big Data
The initial definition of Big Data revolves around the three R's:
- Volume: Large sample size, potentially high-dimensional. Use MapReduce,
Hadoop, etc. when data is too large to be stored in one machine.
- Velocity: Data is generate dand collected very quickly. Increases computation
efficiency.
- Variety: The data types you mention all take different shapes. How to deal
with high-dimensional data, e.g. profiles, images, videos, etc.

### High Dimensional Data
High dimensional data is defined as a data set with a large number of attributes.
Consider Signals, Images, Videos, Surveys.

### High-Dimensional Data vs Big Data 
|         | Small n                                                            | Large n                                            |
|---------|--------------------------------------------------------------------|----------------------------------------------------|
| Small p | Traditional Statistics with limited samples                        | Classic large sample theory **Big Data Challenge** |
| Large p | HD Statistics and optimization **High-Dimensional Data Challenge** | Deep Learning and  Deep Neural Networks            |

### Curse of Dimensionality
**HD Analytics Challenge** is mainly related to the curse of dimensionality.
As distance between observations increases with the dimensions, the
sample size required to learn a model drastically increases.

- Solutions: Feature extractaion and dimension reduction through
low-dimensionality learning.

### Low-Dimensional Learning from High-Dimensional Data
- High-Dimensional data can usually have low dimensional structure
- Real data highly concentrated on low-dimensional, sparse, or degenerate
structure in high-dimensional space.

### LD Learning Methods in this Course
#### Functional Data Analysis
- Splines
- Smoothing Splines
- Kernels
#### Tensor Analysis
- Multilinear Algebra
- Low Rank Tensor Decomposition
#### Rank Deficient Methods
- (Functional) Principal Component Analysis (FPCA)
- Robust PCA (RPCA)
- Matrix Completion

### Functional Data
Definition: A fluctuating quantity or impulse whose variations represent information
and is often represented as a function of time or space. 

Consider Single-channel signals, Multi-channel signals, Images, Point cloud

## Review of Regression
### Linear Regression Models and Least Squares
A linear regression model assumes that the regression function
$E(Y|X)$ is linear in the inputs $X_1, \ldots , X_p$.

We have an input vector $X^T = (X_1, X_2, \ldots, X_p)$ and we want to predict
a real-valued output $Y$. The Linear Regression model has the form:
$$
f(X) = \beta_0 + \sum_{j=1}^p X_j\beta_j.
$$

The most popular estimation technique is to use *least squares* to
pick coefficients $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T$ to minimize
the residual sum of squares
$$
\begin{align}
    \text{RSS}(\beta) &= \sum_{i=1}^N(y_i - f(x_i))^2 \\
    &= \sum_{i=1}^N\left( y_i - \beta_0 - \sum_{j=1}^p x_{ij}\beta_j \right)^2.
\end{align}
$$

How do we minimize $RSS(\beta)$?
$$
\begin{align}
    \text{RSS}(\beta) &= (y - \mathbf{X}\beta)^T(y-\mathbf{X}\beta) \\
    \frac{\partial \text{RSS}}{\partial \beta} 
        &= -2\mathbf{X}^T(y-\mathbf{X}\beta) \\
    \frac{\partial^2 \text{RSS}}{\partial \beta \partial \beta^T} &= 2X^T X
\end{align}
$$
Assuming that $X$ has full column rank, hence $X^TX$ is positive definite, we
set the first derivative to zero.
$$
\begin{align}
    \mathbf{X}^T(y-X\beta) = 0 \\
    \hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^Ty
\end{align}
$$

In order to identify the sampling properties of $\hat{\beta}$, we assume
that the observations of $y_i$ are uncorrelated and have constant variance
$\sigma^2$, and that the $x_i$ are fixed (non-random). The variance-covariance
matrix of the least squares paramter estimates is given by:
$$
    \text{Var}(\hat{\beta}) = (\mathbf{X}^T\mathbf{X})^{-1} \sigma^2
$$
Typically on estimates the variance $\sigma^2$ by
$$
    \hat{\sigma}^2 = \frac{1}{N-p-1}\sum_{i=1}^N (y_i - \hat{y}_i)^2.
$$
The $N-p-1$ rather than $N$ in the denominator makes $\hat{\sigma}^2$ an 
unbiased estimate.
Assume that linear regression is the correct model for the mean;
the conditional expection of $Y$ is linear in $X_1, \ldots, X_p$. We also
Assume the deviations of $Y$ around its expectations are additive and
Gaussian. Therefore:
$$
\begin{align}
    Y &= \text{E}(Y|X_1, \ldots, X_p) + \epsilon \\
    &= \beta_0 + \sum_{j=1}^pX_j \beta_j + \epsilon,
\end{align}
$$
where the error $\epsilon$ is a Gaussian random variable, 
$\epsilon \sim N(0, \sigma^2)$
