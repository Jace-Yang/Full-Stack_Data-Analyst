# Quantifying statistical dependence

```{math}
\newcommand\indep{\perp\kern-5pt\perp}
```


As we saw earlier, two random variables may be *uncorrelated* (the covariance of two random variables may be zero), but that does not imply the two variables are independent. 
This figure from the wikipedia article on [Correlation and Dependence](http://en.wikipedia.org/wiki/Correlation_and_dependence) is a good illustration. The bottom row shows examples of two variables that are uncorrelated, but not statistically independent (eg. we can't factorize the joint $p(X,Y)$ as $p(X)p(Y)$). 

<p><a href="https://commons.wikimedia.org/wiki/File:Correlation_examples2.svg#/media/File:Correlation_examples2.svg"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Correlation_examples2.svg/500px-Correlation_examples2.svg.png" width="80%" alt="Correlation examples2.svg">
</a></p>

So how can we quantify if and two what degree two variables are statistically dependent? 

## Mutual Information

The [**Mutual information**](https://en.wikipedia.org/wiki/Mutual_information) is 
of two random variables is a measure of the mutual dependence between the two variables. It quantifies the "amount of information"  obtained about one random variable through observing the other random variable. The concept of mutual information is intimately linked to that of entropy of a random variable, a fundamental notion in information theory that quantifies the expected "amount of information" held in a random variable.[^footnote1] 

```{important} The **mutual information** $I(X;Y)=0$ *if and only if* $X \indep Y$.

```


The mutual information of two jointly discrete random variables 
$X$ and $Y$ is calculated as a double sum

$$
{\displaystyle \operatorname {I} (X;Y)=\sum _{y\in {\mathcal {Y}}}\sum _{x\in {\mathcal {X}}}{p_{(X,Y)}(x,y)\log {\left({\frac {p_{(X,Y)}(x,y)}{p_{X}(x)\,p_{Y}(y)}}\right)}},}
$$ 
 
 where ${\displaystyle p_{(X,Y)}}$ is the joint probability mass function of $X$ and $Y$ and $p_{X}$ and $p_Y$ are the marginal probability mass functions.$X$ and $Y$ respectively.

In the case of jointly continuous random variables, the double sum is replaced by a double integral

$$
{\displaystyle \operatorname {I} (X;Y)=\int _{\mathcal {Y}}\int _{\mathcal {X}}{p_{(X,Y)}(x,y)\log {\left({\frac {p_{(X,Y)}(x,y)}{p_{X}(x)\,p_{Y}(y)}}\right)}}\;dx\,dy,}
$$

where ${\displaystyle p_{(X,Y)}}$ is now the joint probability density function and $p_{X}$ and $p_Y$ are the marginal probability density functions.

If the log base 2 is used, the units of mutual information are bits.

An equivalent formulation is 

$$
{\displaystyle I(X;Y)=D_{\mathrm {KL} }(P_{(X,Y)}\|P_{X}\otimes P_{Y})}
$$

where 
$D_{{{\mathrm  {KL}}}}$ is the [Kullback–Leibler](https://en.wikipedia.org/wiki/Kullback–Leibler_divergence) divergence, which we will return to later in the course. Here we see that it is the KL distance between the joint and the product of the two marginals, and so it is only zero if the those are identical, which is equivalent to saying $p(X,Y)= p(X)p(Y)$, which is the definition of independence. 

Another useful identity is:

$$
{\displaystyle {\begin{aligned}\operatorname {I} (X;Y)&{}\equiv \mathrm {H} (X)-\mathrm {H} (X|Y)\\&{}\equiv \mathrm {H} (Y)-\mathrm {H} (Y|X)\\&{}\equiv \mathrm {H} (X)+\mathrm {H} (Y)-\mathrm {H} (X,Y)\\&{}\equiv \mathrm {H} (X,Y)-\mathrm {H} (X|Y)-\mathrm {H} (Y|X)\end{aligned}}}
$$

where ${\displaystyle \mathrm {H} (X)}$ and ${\displaystyle \mathrm {H} (Y)}$ are the marginal [entropies](https://en.wikipedia.org/wiki/Information_entropy), 
${\displaystyle \mathrm {H} (X|Y)}$ and ${\displaystyle \mathrm {H} (Y|X)}$ are the [conditional entropies](https://en.wikipedia.org/wiki/Conditional_entropy), and 
${\displaystyle \mathrm {H} (X,Y)}$ is the [joint entropy](https://en.wikipedia.org/wiki/Joint_entropy) of $X$ and $Y$.

```{note} The mutual information is symmetric $I(X;Y)=I(Y;X)$ and non-negative $I(X;Y)\ge 0$.

```

## Distance Correlation

[Distance Correlation](https://en.wikipedia.org/wiki/Distance_correlation)  is a measure of dependence between two paired random vectors of arbitrary, not necessarily equal, dimension. 
Thus, distance correlation measures both linear and nonlinear association between two random variables or random vectors. This is in contrast to Pearson's correlation, which can only detect linear association between two random variables [^footnote2].

```{important} The **distance correlation** is zero  *if and only if* $X \indep Y$.

```

[^footnote1]: Adapted from [https://en.wikipedia.org/wiki/Mutual_information](https://en.wikipedia.org/wiki/Mutual_information)

[^footnote2]: Adapted from [https://en.wikipedia.org/wiki/Distance_correlation](https://en.wikipedia.org/wiki/Distance_correlation)