# Conditonal Probability

Let us start with a graphical introduction to the notion of conditional probability [^footnote1]. 
Imagine you are throwing darts, and the darts  uniformly hit the rectangular dartboard below.

```{figure} ./assets/prob_cousins.png
:name: prob_cousins
:width: 50%

A visual representation of events $A$ and $B$ in a larger sample space $\Omega$ [^footnote1].
```

The dark board has two oval shaped pieces of paper labeled $A$ and $B$. We can graphically convey the probability of hitting $A$ and the probability of hitting $B$ with the images below.

```{figure} ./assets/pA_and_pB.png
:name: cousins_and

A visual representation of $P(A)$ and $P(B)$ [^footnote1].
```

And we can also talk about the probability of hitting $A$ **and** $B$, which is often written as $A \cap B$, as the image below.

```{figure} ./assets/pAandB.png
:name: cousins_and

A visual representation of $P(A \cap B)$ [^footnote1].
```

In both cases the denominator is the full entire sample space $\Omega$ (the rectangle).

Now let's consider the **conditional probability** $P(A \mid B)$, which is said "probability of $A$ **given** $B$". We know that the dart hit $B$, so the denominator is no longer the entire sample space $\Omega$ (the rectangle). Instead, it the denominator is $B$. Similarly, the numerator is no longer all of $A$, because some parts of $A$ aren't also in $B$. Instead, the numertor is the intersection $A \cap B$. We can visualize this as:

```{figure} ./assets/conditional.png
:name: cousins_conditional

A visual representation of $P(A \mid B)$ [^footnote1].
```

We will extend this visual representation in the section on [Bayes' Theorem](./bayes_theorem).




## Visualizing conditional distributions for continuous data

Consider the arbitrary joint distribution $p_{XY}(X,Y)$ shown below

```{figure} ./assets/schematic_p_xy.png
:name: schematic_p_x_given_y

A schematic of the joint $p(X,Y)$
```

If we want to condition on the random varaible $Y$ taking on the value $y=-1.15$, then the conditional distribution $p_{X\mid Y}(X|Y)$ is just a normalized version of a slice through the joint:

$$
p_{X\mid Y}(X \mid Y=y) = \frac{p_{XY}(X,Y=y)}{\int p_{XY}(x,y) dx} =  \frac{p_{XY}(X,Y=y)}{p_Y(Y=y)}
$$(conditional_x_given_y)

```{figure} ./assets/schematic_p_x_given_y.png
:name: schematic_p_x_given_y

A schematic of the slice through the joint $p(X,Y=y)$ and the normalized conditional $p(X|Y)$.
```

Similarly, if we want to condition on the random varaible $X$ taking on the value $x=1.75$, then the conditional distribution $p_{Y\mid X}(Y|X)$ is just a normalized version of a slice through the joint:

$$
p_{Y\mid X}(Y \mid X=x) = \frac{p_{XY}(X=x,Y)}{\int p_{XY}(x,y) dy} =  \frac{p_{XY}(X=x,Y)}{p_X(X=x)}
$$(conditional_y_given_x)

```{figure} ./assets/schematic_p_y_given_x.png
:name: schematic_p_y_given_x

A schematic of the slice through the joint $p(X=x,Y)$ and the normalized conditional $p(Y|X)$.
```

```{note}
Here's a [link to the notebook](correlation_schematic.ipynb) I used to make these images in case it is useful.

```

## Marginal Distributions

The normalization factors in the denominator of Equations {eq}`conditional_x_given_y` and {eq}`conditional_y_given_x` involve probability distributions over an individual variables $p_X(X)$ or $p_Y(Y)$ without conditioning on the other. These are called **marginal distributions** and they correspond to integrating out (or *marginalizing*) the other variable(s). Eg. 

$$
p_X(x) = \int  p_{XY}(x,y) dy
$$(marginalization_over_y)

In many ways, marginalization is the opposite of conditioning. 

For high dimensional problems, marginalization is difficult as it involves high dimensional integrals. Naive numerical integration is often not tractable, which has motivated a number of different approaches to approximate the integrals, such as Monte Carlo integration. 


## Chain Rule of Probability

One very powerful and useful result is that, without loss of generality, one can decompose a joint distribution into the appropriate product of conditionals. For example, one can always write the joint distribution for $X$ and $Y$ as

$$
p(X,Y) = p(X|Y) p(Y)
$$

Similarly, one can always decompose the the joint for three variables as

$$
p(X,Y,Z) = p(X|Y,Z) p(Y|Z) p(Z)
$$

And this type of decomposition for the joint for $N$ random variables $X_1, \dots, X_N$ is often written in this way:

$$
p_N(X_1, \dots, X_N) = \prod_{i=2}^N p_i(X_i|X_{i-1}, \dots X_{1}) p_1(X_{1})
$$

Note that here I've added subscripts to the distributions as they are all different in general. In some cases, one uses this same kind of decomposition and additionally assumes that there is some structure across the different distributions (eg. in a Markov process or an autoregressive model).

An alternative notation that is often found is:

$$
p(X_1, \dots, X_N) = \prod_{i=1}^N p(X_i|X_{< i})
$$

where the first term $p(X_{1})$ without any conditioning is implied.


### A more general formulation

The formulation of the chairn rule 
See [Theorem 1.2.2 (Chain rule) in the NYU CDS lecture notes on Probability and Statistics](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf) for a more general formulation 

## A mnemonic on conditional distributions: Units

We will see many different types of conditional distributions in this course, and manipulating them can be error prone and confusing. Manipulating conditional distributions takes some practice, it is not much different than learning to manipulate upper- and lower-indices in special realtivity and Einstein notation. As we will see later, some distributions have additional structure -- some variables may be (assmed to be) independent or conditionally independent -- and in these cases the decomposition isn't completely general, but it there are still some rules. 

For example, I know that $p(X,Y|Z)p(X)$ is not a valid decomposition of any joint $p(X,Y,Z)$ or conditional $p(X,Y|Z)$. I know this immediately by inspection because the $X$ appears on the left of the $\mid$ more than once. If $X,Y,Z$ are continuous and have units, then the units of this expression would be $[Y]^{-1}[X]^{-2}$. Similarly, if I wanted to check that it was normalized I would want to integrate it. While I can assume $\int  p(x,y|z) dx dy= 1$ and $\int p(x) dx = 1$, there is no reason for $\int  p(x,y|z)p(x)$ will be 1, and it will still have units of $[X]^{-1}$.

Personally, I like to sort the terms like this $p(X,Y) = p(X|Y) p(Y)$ instead of like this $p(X,Y) = p(Y) p(X|Y)$. Or like this $P(A \cap B) = P(A \mid B) p(B)$ instead of like this $P(A \cap B) = p(B) P(A \mid B)$. In both cases, what one can form a joint distribution by starting with a conditioanl and then multiplying by a distribution for what is being conditioned on. I find that putting the terms in this order helps me avoid mistakes and it's easier to connect to the chian rule of probability.

### Exercise

Which of the following are valid (not necessarily general) decompositions of some probability distribution?

<label><input type="checkbox" id="box-conditional-1" class="box"> $p(W,X,Y|Z)P(Z)$ </input></label>

<label><input type="checkbox" id="box-conditional-2" class="box"> $p(W,X,Y|Z)P(Y)$ </input></label>

<label><input type="checkbox" id="box-conditional-3" class="box"> $p(W|X,Y,Z)P(X,Y,Z)$ </input></label>

<label><input type="checkbox" id="box-conditional-4" class="box"> $p(W|X,Y,Z)P(X)$ </input></label>

<label><input type="checkbox" id="box-conditional-5" class="box"> $p(W|X,Y,Z)P(W,Y,Z)$ </input></label>

<label><input type="checkbox" id="box-conditional-6" class="box"> $p(W,X)p(Y,Z)$ </input></label>

<label><input type="checkbox" id="box-conditional-7" class="box"> $p(W|X)p(Y,Z)$ </input></label>

<label><input type="checkbox" id="box-conditional-8" class="box"> $p(W,Y)p(X|Z)$ </input></label>

<label><input type="checkbox" id="box-conditional-9" class="box"> $p(W|X,Y,Z)p(X,Y|Z)p(Z)$ </input></label>


[^footnote1]: These images are adapted from lectures by Bob Cousins.
