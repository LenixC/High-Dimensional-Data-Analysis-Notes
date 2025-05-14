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
#### Regression
A linear regression model assumes that the regression function
$E(Y|X)$ is linear in the inputs $X_1, \ldots , X_p$.

We have an input vector $X^T = (X_1, X_2, \ldots, X_p)$ and we want to predict
a real-valued output $Y$. The Linear Regression model has the form:

```math
f(X) = \beta_0 + \sum_{j=1}^p X_j\beta_j.
```

#### Regression - Least Squares Estimate
The most popular estimation technique is to use *least squares* to
pick coefficients $\beta = (\beta_0, \beta_1, \ldots, \beta_p)^T$ to minimize
the residual sum of squares

```math
\begin{align}
    \text{RSS}(\beta) &= \sum_{i=1}^N(y_i - f(x_i))^2 \\
    &= \sum_{i=1}^N\left( y_i - \beta_0 - \sum_{j=1}^p x_{ij}\beta_j \right)^2.
\end{align}
```

How do we minimize $RSS(\beta)$?

```math
\begin{align}
    \text{RSS}(\beta) &= (y - \mathbf{X}\beta)^T(y-\mathbf{X}\beta) \\
    \frac{\partial \text{RSS}}{\partial \beta} 
        &= -2\mathbf{X}^T(y-\mathbf{X}\beta) \\
    \frac{\partial^2 \text{RSS}}{\partial \beta \partial \beta^T} &= 2X^T X
\end{align}
```

Assuming that $X$ has full column rank, hence $X^TX$ is positive definite, we
set the first derivative to zero.

```math
\begin{align}
    \mathbf{X}^T(y-X\beta) = 0 \\
    \hat{\beta} = (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^Ty
\end{align}
```

In order to identify the sampling properties of $\hat{\beta}$, we assume
that the observations of $y_i$ are uncorrelated and have constant variance
$\sigma^2$, and that the $x_i$ are fixed (non-random). The variance-covariance
matrix of the least squares paramter estimates is given by:
```math
\text{Var}(\hat{\beta}) = (\mathbf{X}^T\mathbf{X})^{-1} \sigma^2
```

### Geometric Interpretation
The Projection Matrix (Hat matrix) 
```math
\hat{y} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1} \mathbf{X}^Ty = \mathbf{H}y
```
orthogonally projects $y$ onto the hyperplane spanned by $\mathbf{X}.$

The projection of $\hat{y}$ represents the vector of predictions by the 
least square method.

### Properties of OLS Estimators
Typically on estimates the variance $\sigma^2$ by
```math
\hat{\sigma}^2 = \frac{1}{N-p-1}\sum_{i=1}^N (y_i - \hat{y}_i)^2.
```

The $N-p-1$ rather than $N$ in the denominator makes $\hat{\sigma}^2$ an 
unbiased estimate.
Assume that linear regression is the correct model for the mean;
the conditional expection of $Y$ is linear in $X_1, \ldots, X_p$. We also
Assume the deviations of $Y$ around its expectations are additive and
Gaussian. Therefore:
```math
\begin{align}
    Y &= \text{E}(Y|X_1, \ldots, X_p) + \epsilon \\
    &= \beta_0 + \sum_{j=1}^pX_j \beta_j + \epsilon,
\end{align}
```

where the error $\epsilon$ is a Gaussian random variable, 
$\epsilon \sim N(0, \sigma^2)$
Therefore,
```math
\hat{\beta} \sim N(\beta, (\mathbf{X}^T\mathbf{X})^{-1}\sigma^2).
```

To test the hypothesis that a particular coefficient $\beta_j = 0$, we
form the standardized coefficient or *Z-score*
```math
z_j = \frac{\hat{\beta}_j}{\hat{\sigma} \sqrt{v_j}}
```

### Feature Extraction Usiong Regression
Consider a high dimensional signal, we can fit the polynomial regression
```math
y = \beta_0 + \beta_{i} t + \beta_2 t^2 + \beta_3 + t^3
```
and extract only the estimated $\hat{\beta}$.

## Splines
### Polynomial Vs. Nonlinear Regression
$m$-th order polynomial regression
```math
y = \beta_0 + \beta_1 x + \beta_2 x^2 + 
        \beta_3 x^3 + \cdots + \beta_m x^m + \epsilon
```
Nonlinear regression often requires domain knowledge or first principles for
finding the underlying nonlinear function. Often, these are piecewise.

### Disadvantages of Polynomial Regression
- Remote part of the function is very sensitive to outliers
- Less flexibility due to global functional structure, you are always trying
to convert a curve to a polynomial
To repair these, we seek to build local strcuture instead of global structure.

### Splines
A spline is a linear combination of piecewise polynomial functions under
continuity construction. We partition the domain of $x$ into continuous intervals
and fit polynomials in each interval separately. This provides flexibility and
local fitting.

Suppose $x \in [a, b]$. Partition the $x$ domain using the following 
points (knots).
```math
a < \xi_1 < \xi_2 < \cdots < \xi_K < b \quad\quad \xi_0 = a, \xi_{K=1} = b
```

Fit a polynomial in each interval under the continuity conditions and 
integrate them by
```math
f(X) = \sum_{m=1}^K \beta_m h_m(X)
```

### Estimation
The least squares method can be used to estimate the coefficients
```math
\mathbf{H} = [h_1(x)\ h_2(x)\ h_3(x)\ h_4(x)\ h_5(x)\ h_6(x)] 
        \rightarrow \hat{\beta} = (\mathbf{H}^T\mathbf{H})^{-1}\mathbf{H}^Ty
```
Linear smoother: 
$\hat{y} = \mathbf{H}\hat{\beta} = \mathbf{H}(\mathbf{H}^T \mathbf{H})^{-1} \mathbf{H}^Ty = \mathbf{S}y$ 

Degrees of Freedom: $df = \text{trace} \mathbf{S}$

Truncated power basis functions are simple and algebraically appealing, 
but not efficient for computation and ill-posed and numerically unstable.
