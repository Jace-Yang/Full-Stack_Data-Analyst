# Random Variables

The basic idea of **random variables** is inuitive and familiar for physicists, and it is perhapse *the* fundamental idea in probabilistic thinking. 
At the same time, randomness is at the heart of some of the deepest mysteries of physics: the transition from the determinism of classical mechanics to indeterminism in quantum mechanics. 
Furthermore, the notation and terminology used by statisticians is often unfamiliar or awkward to physicists and the rigorous mathematical treatment of random variables may seem overly formal and opaque. 

```{note}
The  [Stanford lectures on Probability and statistics](http://cs229.stanford.edu/section/cs229-prob.pdf) and the  [NYU CDS lecture notes on Probability and Statistics](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf) both start from the formal definition of Probability Spaces, but let's start with something a little more intuitive. 
```


To start with we will make the distinction between two types of random variables:
 * **Discrete random variables** : e.g. the flip of a coin, the roll of a die, the number of decays of a radioactive substance in a fixed time interval, etc.
 * **Continuous random variables** : e.g. the height of a person, the mass of a star, the time interval between two subsequent radioactive decays, etc.

 In both cases we have in mind the notion of an underlying **population** and the particular values that different instances (or **realizations**) that these random values may take. The realizations are random draws from some population: e.g. the height of a particular person drawn from a population of people, the mass of a particular star drawn from a population of stars, the result of a particular flip of a coin drawn from a (potentially hypothetical) population of coin flips. Consider this quote from the  [NYU CDS lecture notes on Probability and Statistics](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf):
 
```{admonition} Notation
:class: note
A random variable quantifies our uncertainty about the quantity it represents, not the value that it happens to finally take once the outcome is revealed. You should *never* think of a random variable as having a fixed numerical value. If the outcome is known, then that determines a realization of the random variable. In order to stress the difference between random variables and their realizations, we denote the former with uppercase letters $(X, Y , . . . )$ and the latter with lowercase letters $(x, y, . . . )$.
```

We often say that the random variable $X$ is **distributed** according to a certain distribution denoted $p_X$.  It is also useful to denote $\mathbb{X}$ for the space that the realizations $x$ live in (eg. natural numbers $\mathbb{N}$, real numbers $\mathbb{R}$, d-dimensional Euclidean space $\mathbb{R}^d$, etc.), In order to refer to the probability (density) that the random variable $X$ takes on the value $x$, we write $p_X(X=x)$ (often shortened to $p_X(x)$ or just $p(x)$ if the context is clear). 

 ```{admonition} Terminology
 Statisticians often links the type of random variable with its distribution (eg. "a Poisson random variable" or "a Gaussian random variable") as opposed to the data type the realization take on (i.e. a natural number or a real number). 
 ````

It is important to make the distinction between the discrete and continous cases:
 * **Probability Mass Function** (pmf) describes the distribution of a discrete random variables (eg. $x\in \mathbb{N}$), and $p_X(x)$ is unitless (or has "units of probability")
 * **Probability Density Functions** (pdf) describes the distribution of a continuous random variable (eg. $x \in \mathbb{R}$), and $p_X(x)$ has units of probability per unit $X$.

 This is analogous to thinking of point masses or point charges in space versus  mass-density or charge-density distributed along a line, surface, or volume. 
 Just as the mass or charge in a region is the integral of this mass-density or charge-density in that region, the probability that a continous $x$ falls in some region $W \in \mathbb{X}$ is $P(x\in W) = \int_W p_X(x) dx$.

 These distributions have a few intuitive properties, which correspond to the [axioms of probability](axioms_of_prob):
  * $\sum_{x} p_X(x) = 1$ or in the continous case  $\int dx p_X(x) = 1$
  * $p_X(x) \ge 0$ for all $x$
  * if $A$ and $B$ are mutually exclusive  (or disjoint so that their intersection is empy, $A \cap B = \emptyset$ ), then $p(A \cup B) = p(A)+p(B)$. For continuous variables, you could write $\int_{A \cup B} p_X(x) dx = \int_{A} p_X(x) dx + \int_{B} p_X(x) dx$

```{note}
In the continous case it is totally fine for the probability density $p_X(x)>1$. Consider a Gaussian distribution with $\sigma = 0.01$. 
```

```{note}
It is somewhat common that probability density functions are denoted $f(X)$ instead of $p(X)$ or to use a capital $P(X)$ to denote probability and a lower-case $p(X)$ to denote a probability density. Usually, this can be sorted out from context. 
```
In terms of notation, it is common to see $X \sim p_X$, which is read as "(the random variable) $X$ is distributed as (the distribution) $p_X$". Sometimes one may also see $X \sim p_X(\cdot)$. This notation really emphasizes $X$ as a random variable and $p_X$ as a distribution, and with this notation it does not make sense to write $x \sim p_X$. However, it is fairly common in some areas of physics to write $p(x)$ to refer to the distribution with the idea that $x$ is the explicit realization of a random variable, but the argument to a function. These notational issues may seem overstated in this document, but it is my experience that it is a barier to physicists reading the statistics literature and a fundamental cause of needless reinvention of the wheel. 

## Cumulative distributions 

A related concepts is the **cumulative distribution function** (cdf) for a real-valued random variable $X$, which is defined as the probability the random variable $X$ is less than or equal to some particular value $x$

$$
F_X(x) := P(X \le x) 
$$

I think that it is inuitive for physicists to think of a proability density function as the fundamental object and to define $F_X(x) = \int_{-\infty}^x p_X(x) dx$; however, typically the formal approach is the opposite and one defines 

$$
p_X(x) := \frac{dF_X}{dx} .
$$

This kind of fine print matters is important formally in cases where the derivative of $F_X(x)$ does not exist, but rarely matters in practice. 

So what about continous multivariate data $x \in \mathbb{R}^d$? How does one define a cumulative distribuiton in that case? The integral "from minus infinity to $x$" doesn't seem to make sense, or at least it is ambiguous. Say we have two continous random variables $X$ and $Y$, then one can define the **joint cumulative distribution function** 

$$
F_{XY}(x,y) := P(X\le x, Y\le y), 
$$

ie. the probability that the random variable $X \le x$ *and* $Y \le y$. Personally, this always bothered me as a physicist because it seems like it is sensitive to an arbitrary choice of axes for my two dimensional data. But mathematically, it works for formally defining what to me is a more natural **joint probability density function**

$$
p_{XY}(x,y) := \frac{\partial^2 F_{XY}(x,y)}{\partial x \partial y}.
$$

The generalizatio to data in $\mathbb{R}^d$ is straight forward with the $d^\textrm{th}$ partial derivative. (Note, at this point in Secton 3.2 of  [NYU CDS lecture notes on Probability and Statistics](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf) the notation for the joint pdf changes to $f_{XY}(x,y)$)

## Futher reading

With this introduction, I invite you to read the [NYU CDS lecture notes on Probability and Statistics](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf) Sections 2.1-2.3, 3.1-3.3. 
As you will find, this requires understanding the notion of a **probability space**, a **sample space**, a **probability measure**, and the mathematical concept of a $\sigma$**-algebra**. These are defined and discussed in Section 1.

You may also be interested in reading about the idea of a [Copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)), which relates the cumulative distribution functions for individual random variables (marginals) $X$ and $Y$ to the joint distribution. 

```{warning}
 The formal treatments of probability spaces makes subtle distinctions between terms like *event*,  *observation*, *sample*, and *outcome*, which physicists may tend to use interchangibly. Furthermore, in causal inference there is a distinction made between *observational studies* and *experiments*. 
 ```

