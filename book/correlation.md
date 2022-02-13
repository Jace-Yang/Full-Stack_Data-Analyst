# Covariance and Correlation

## Variance for a single variable
The expected value or mean of a random variable is the first moment, analogous to a center of mass for a rigid body. The **variance** of a single random variable is the second moment:  it is the expectation of the squared deviation of a random variable from its mean. It is analogous to the moment of inertia about the center of mass.

$$
\operatorname{Var} (X)=\mathbb{E} \left[(X-\mu )^{2}\right] = \int (x-\mu)^2 p(x) dx,
$$
where $\mu = \mathbb{E}[X]$

The units of $\operatorname{Var} (X)$  are $[\operatorname{Var} (X)] = [X]^2$. For that reason, it is often more intuitive to work with the **standard deviation** of $X$, usually denoted $\sigma_X$, which is the square root of the variance:

$$
\sigma_X^2 = \operatorname{Var} (X)
$$

In statistical mechanics, you may have seen notation like this: $\sigma_X = \sqrt{ \left\langle \left( X - \langle X \rangle \right)^2 \right\rangle }$

## Covariance   

When dealing with multivariate data, the notion of variance must be lifted to the concept of **covariance**. Covariance captures how one variable deviates from its mean as another variable deviates from it's mean. Say we have two variables $X$ and $Y$, then the covariance for the two variables is defined as

$$
\textrm{cov} (X,Y)=\mathbb{E} {{\big [}(X-\mathbb{E} [X])(Y-\mathbb{E} [Y]){\big ]}}
$$(covariance)

If $X$ is on average greater than its mean when $Y$ is greater than its mean (and, similarly, if $X$ is on average less than its mean when $Y$ is less than its mean), then we say the two variables are **positively correlated**. In the opposit case,  when $X$ is on average less than its mean when $Y$ is greater than its mean (and vice versa), then we say the two variables are **negatively correlated**. If $\operatorname{Cov}(X,Y) = 0$, then we say the two variables are **uncorrelated**.

A useful identity

$$
{\displaystyle {\begin{aligned}\textrm{cov} (X,Y)&=\mathbb {E} \left[\left(X-\mathbb {E} \left[X\right]\right)\left(Y-\mathbb {E} \left[Y\right]\right)\right]\\&=\mathbb {E} \left[XY-X\mathbb {E} \left[Y\right]-\mathbb {E} \left[X\right]Y+\mathbb {E} \left[X\right]\mathbb {E} \left[Y\right]\right]\\&=\mathbb {E} \left[XY\right]-\mathbb {E} \left[X\right]\mathbb {E} \left[Y\right]-\mathbb {E} \left[X\right]\mathbb {E} \left[Y\right]+\mathbb {E} \left[X\right]\mathbb {E} \left[Y\right]\\&=\mathbb {E} \left[XY\right]-\mathbb {E} \left[X\right]\mathbb {E} \left[Y\right],\end{aligned}}}
$$


## Correlation coefficient

The covariance $\operatorname{Cov}(X,Y)$ has units $([X][Y])^{-1}$, and thus depends on the units for $X$ and $Y$. It is desireable to have a unitless measure of how "correlated" the two variables are. One way to do this is through the [**Correlation coefficient**](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) $\displaystyle \rho _{X,Y}$, which simply divides out the standard deviation of $X$ and $Y$ 

$$
{\displaystyle \rho _{X,Y}={\frac {\textrm{cov} (X,Y)}{\sigma _{X}\sigma _{Y}}}},
$$(correlation_coefficient)

where $\sigma_X^2 = \textrm{cov}(X,X)$ and $\sigma_Y^2 = \textrm{cov}(Y,Y)$

```{warning}

It is common to mistakenly think that if two variables $X$ and $Y$ are "uncorrelated" that they are [statistically independent](./independence), but this is not the case.
It is true that if two variables  $X$ and $Y$ are "correlated" (have non-zero covariance), then the two variables are [statistically dependent](./independence), but the converse is not true in general.
We will see this in our [Simple Data Exploration](datasaurus-long).
```

## Covariance matrix

When dealing with more than two variables, there is a straightforward generalization of covariance (and correlation) in terms of a **covariance matrix** [^footnote1]. Given random variables $X_1, \dots, X_N$, the covariance matrix is an $N\times N$ matrix whose $(i,j)$ entry is the covariance

$$
{\displaystyle \operatorname {K} _{X_{i}X_{j}}=\operatorname {cov} [X_{i},X_{j}]=\mathbb{E} [(X_{i}-\mathbb{E} [X_{i}])(X_{j}-\mathbb{E} [X_{j}])]}
$$

If the entries are represented as a column vector ${\displaystyle \mathbf {X} =(X_{1},X_{2},...,X_{n})^{\mathrm {T} }}$,  then the covariance matrix can be written as

$$
{\displaystyle \operatorname {K} _{\mathbf {X} \mathbf {X} }=\operatorname {cov} [\mathbf {X} ,\mathbf {X} ]=\mathbb{E} [(\mathbf {X} -\mathbf {\mu _{X}} )(\mathbf {X} -\mathbf {\mu _{X}} )^{\rm {T}}]=\mathbb{E} [\mathbf {X} \mathbf {X} ^{T}]-\mathbf {\mu _{X}} \mathbf {\mu _{X}} ^{T}}
$$

with ${\displaystyle \mathbf {\mu _{X}} =\mathbb{E} [\mathbf {X} ]}$ also represented as a column vector.

```{note}
The inverse of this matrix, 
${\displaystyle \operatorname {K} _{\mathbf {X} \mathbf {X} }^{-1}}$, if it exists, is also known as the **concentration matrix** or **precision matrix**.
```

## Correlation Matrix

An entity closely related to the covariance matrix is the **correlation matrix** [^footnote1], 

$$
{\displaystyle \operatorname {corr} (\mathbf {X} )={\begin{bmatrix}1&{\frac {\mathbb{E} [(X_{1}-\mu _{1})(X_{2}-\mu _{2})]}{\sigma (X_{1})\sigma (X_{2})}}&\cdots &{\frac {\mathbb{E} [(X_{1}-\mu _{1})(X_{n}-\mu _{n})]}{\sigma (X_{1})\sigma (X_{n})}}\\\\{\frac {\mathbb{E} [(X_{2}-\mu _{2})(X_{1}-\mu _{1})]}{\sigma (X_{2})\sigma (X_{1})}}&1&\cdots &{\frac {\mathbb{E} [(X_{2}-\mu _{2})(X_{n}-\mu _{n})]}{\sigma (X_{2})\sigma (X_{n})}}\\\\\vdots &\vdots &\ddots &\vdots \\\\{\frac {\mathbb{E} [(X_{n}-\mu _{n})(X_{1}-\mu _{1})]}{\sigma (X_{n})\sigma (X_{1})}}&{\frac {\mathbb{E} [(X_{n}-\mu _{n})(X_{2}-\mu _{2})]}{\sigma (X_{n})\sigma (X_{2})}}&\cdots &1\end{bmatrix}}.}
$$

Each element on the principal diagonal of a correlation matrix is the correlation of a random variable with itself, which always equals 1.

Equivalently, the correlation matrix can be written in vector-matrix form as

$$
{\displaystyle \operatorname {corr} (\mathbf {X} )={\big (}\operatorname {diag} (\operatorname {K} _{\mathbf {X} \mathbf {X} }){\big )}^{-{\frac {1}{2}}}\,\operatorname {K} _{\mathbf {X} \mathbf {X} }\,{\big (}\operatorname {diag} (\operatorname {K} _{\mathbf {X} \mathbf {X} }){\big )}^{-{\frac {1}{2}}},}
$$

where 
${\displaystyle \operatorname {diag} (\operatorname {K} _{\mathbf {X} \mathbf {X} })}$ is the matrix of the diagonal elements of 
${\displaystyle \operatorname {K} _{\mathbf {X} \mathbf {X} }}$ (i.e., a diagonal matrix of the variances of 
$X_{i}$ for $i=1,\dots ,n)$.



### Visualizing covariance as an ellipse

Often an ellipse is used to visualize a covariance matrix, but why? This is only well-motivated if one expects the data to be normally distributed (aka Gaussian distributed). This is because the contours of a 2-d normal are ellipses, and in higher dimensions the contours are ellipsoids. 


```{figure} ./assets/001_vanilla_ellipse.png
width: 30%

A scatter plot of two correlated, normally-distributed variables and the error ellipse from [*An Alternative Way to Plot the Covariance Ellipse* by Carsten Schelp](https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html).
```

Consider a random variable $X$ that is distributed as a multivariate normal (aka multivariate Gaussian) distribution, e.g.  ${\displaystyle \mathbf {X} \ \sim \ {\mathcal {N}}({\boldsymbol {\mu }},\,{\boldsymbol {\Sigma }}})$, where $\boldsymbol{\mu}$ is the multivariate mean and $\Sigma$ is the covariane matrix. The probability density for the multivariate normal is given by

$$
\displaystyle p_{\mathbf {X} }(x_{1},\ldots ,x_{k} | \boldsymbol {\mu }, {\boldsymbol {\Sigma })=
{\frac {\exp \left(-{\frac {1}{2}}({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }})\right)}{\sqrt {(2\pi )^{k}|{\boldsymbol {\Sigma }}|}}}}
$$

The contours correspond to values of $\mathbf{X}$ where $({\mathbf {x} }-{\boldsymbol {\mu }})^{\mathrm {T} }{\boldsymbol {\Sigma }}^{-1}({\mathbf {x} }-{\boldsymbol {\mu }}) = \textrm{Constant}$.


Understanding the geometry of this ellipse requires the linear algebra of the covariance matrix, and it's a useful excercise to go through:
 * [This notebook](./covariance_ellipse) is duplicated from the repository linked to in this article: [*An Alternative Way to Plot the Covariance Ellipse* by Carsten Schelp](https://carstenschelp.github.io/2018/09/14/Plot_Confidence_Ellipse_001.html), which has a GPL-3.0 License. 
  * This is also a nice [page](https://cookierobotics.com/007/)


## With empirical data

We can estimate the covariance of the parent distribution $p_{XY}$ with the sample covariance, using the sample mean in place of the  expectation $\mathbb{E}_{p_X}$. 


[^footnote1]: Adapted from [Wikipedia article on Covariance Matrix](https://en.wikipedia.org/wiki/Covariance_matrix)

As we will see in our [Simple Data Exploration](datasaurus-long) and [Visualizing joint and marginal distributions](distributions/visualize_marginals), the sample covariance and correlation matrices can be conveniently computed for a `pandas` dataframe with `dataframe.cov()` and `dataframe.corr()`