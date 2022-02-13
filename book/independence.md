# Independence
```{math}
\newcommand\indep{\perp\kern-5pt\perp}
```

As discussed in the previous section, **conditional probabilities** quantify the extent to which the knowledge of the occurrence of a certain event affects the probability of another event [^footnote1].
In some cases, it makes no difference: the events are independent. More formally, events $A$ and $B$ are **independent** if and only if

$$
P (A|B) = P (A) .
$$

This definition is not valid if $P (B) = 0$. The following definition covers this case and is otherwise
equivalent.

```{admonition} Definition (Independence). 
Let $(\Omega,\mathcal{F},P)$ be a probability space. Two events $A,B \in \mathcal{F}$
are independent if and only if

$$
P (A \cap B) = P (A) P (B) .
$$
```
```{admonition} Notation
This is often denoted $ A \indep B $
```

Similarly, we can define **conditional independence** between two events given a third event.
$A$ and $B$ are conditionally independent given $C$ if and only if

$$
P (A|B, C) = P (A|C) ,
$$

where $P (A|B, C) := P (A|B \cap C)$. Intuitively, this means that the probability of $A$ is not affected by whether $B$ occurs or not, as long as $C$ occurs. 

```{admonition} Notation
This is often denoted $ A \indep B \mid C$
```

## Graphical Models

There is a graphical model representation for joint distributions $P(A,B,C)$ that encodes their conditional (in)dependence known as a **probabilistic graphical model**. For this situation $ A \indep B \mid C$, the graphical model looks like this:

<img src="./assets/AperpBmidC.png" />

The lack of an edge directly between $A$ and $B$ indicates that the two varaibles are conditionally independent. This image was produced with `daft`, and there are more examples in [Visualizing Graphical Models](./pgm/daft).

[^footnote1]: This text is based on excerpts from Section 1.3 of [NYU CDS lecture notes on Probability and Statistics](https://cims.nyu.edu/~cfgranda/pages/stuff/probability_stats_for_DS.pdf)
