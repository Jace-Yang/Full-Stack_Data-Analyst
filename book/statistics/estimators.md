# Estimators

One of the main differences between topics of probability and topics in statistics is that in statistics we have some task in mind. 
While a probability model $P_X(X \mid \theta)$ is an object of study when discussing probability, in statistics we usually want to
*do* something with it. 

The first example that we will consider is to estimate the true, unknown value $\theta^*$ given some dataset $\{x_i\}_{i=1}^N$ 
assuming that the data were drawn from $X_i \sim p_X(X|\theta^*)$.

```{admonition} Definition
An estimator $\hat{\theta}(x_1, \dots, x_N)$ is a function of the data (that aims to estimate the true, unknown value $\theta^*$ assuming that the data were drawn from $X_i \sim p_X(X|\theta^*)$.
```

There are several concrete estimators for different quantities, but this is an abstract definition of what is meant by an estimator. It is useful to think of the estimator as a procedure that you apply to the data, and then you can ask about the properties of a given procedure.


```{admonition} Terminology
These closely related terms have slightly different meanings: 
 * The *estimand* refers to the parameter $\theta$ being estimated.
 * The *estimator* refers to the function or procedure $\hat{\theta}(x_1, \dots, x_N)$
 * The specific value that an estimator takes (returns) for specific data is known as the *estimate*.
```

We already introduced two estimators when studying [Transformation properties of the likelihood and posterior](.distributions/invariance-of-likelihood-to-reparameterizaton.html#equivariance-of-the-mle):
 * The maximum likelihood estimator: $\hat{\theta}_\textrm{MLE} := \textrm{argmax}_\theta p(X=x \mid \theta)$
 * The maximum a posteriori estimator: $\hat{\theta}_{MAP} := \textrm{argmax}_\theta p(\theta \mid X=x)$

Note both of these estimators are defined by procedures that you apply once you have specific data.


```{admonition} Notation
The estimate $\hat{\theta}(X_1, \dots, X_N)$ depends on the random variables $X_i$, so it is itself a random variable (unlike the parameter $\theta$).
Often the estimate is denoted $\hat{\theta}$ and the dependence on the data is implicit. 
Subscripts are often used to indicate which estimator is being used, eg. the maximum likelihood estimator $\hat{\theta}_\textrm{MLE}$ and the maximum a posteriori estimator $\hat{\theta}_\textrm{MAP}$.
```

```{hint}
It is often useful to consider two straw man estimators:
 * A constant estimator: $\hat{\theta}_\textrm{const} = \theta_0$ for $\theta_0 \in \Theta$
 * A random estimator: $\hat{\theta}_\textrm{random} =$ some random value for $\theta$ independent of the data
Neither of these are useful estimators, but they can be used to help clarify your thinking due to their obvious properties. 
```
