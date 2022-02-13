# Empirical Distribution

Often we are working directly with data and we don't know the parent distribution that generated the data. 

We often denote a dataset with $N$ data points indexed by $i$ as $\{x_i\}_{i=1}^N$. 

Sometimes this dataset is thought of a samples or realizatiosn from some parent distribution. For instance, we often assume that we have **independent and identically distributed (iid)** data $x_i \sim p_X$ for $i=1\dots N$. 

In other cases one thinks of this data set as an **emperical distribution**

$$
p_\textrm{emp, X} = \frac{1}{N} \sum_{i=1}^N \delta(x-x_i)
$$


