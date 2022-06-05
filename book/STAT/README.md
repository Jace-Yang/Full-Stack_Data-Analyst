# 基础

### Non-negative matrix factorization (NMF) 

一种利用矩阵因子化将高维离散矩阵分解为两个低维矩阵，从而实现降维、分层的算法

### 各种回归的形式

|  | Functional Form | Marginal Effect | Elasticity |
| :--- | :--- | :--- | :--- |
| Linear | Y=beta_(0)+beta_(1)X | beta_(1) | beta_(1)X//Y |
| Linear-log | Y=beta_(0)+beta_(1)ln X | beta_(1)//X | beta_(1)//Y |
| Quadratic | Y=beta_(0)+beta_(1)X+beta_(2)X2 | beta_(1)+2beta_(2)X | (beta_(1)+2beta_(2)X)X//Y |
| Log-linear | ln Y=beta_(0)+beta_(1)X | beta_(1)Y | beta_(1)X |
| Double-log | ln Y=beta_(0)+beta_(1)ln X | beta_(1)Y//X | beta_(1) |
| Logistic | ln[Y//(1-Y)]=beta_(0)+beta_(1)X | beta_(1)Y(1-Y) | beta_(1)(1-Y)X |

- Elasticity
    $$
    \eta_{y, x}=\frac{\% \Delta Y}{\% \Delta X}=\frac{\Delta Y / Y}{\Delta X / X}=\frac{\Delta Y}{\Delta X} \frac{X}{Y}
    $$
    \% change in $Y$ with respect to a $\%$ change in $X$ for a small change in $X$

- Marginal effect $=\Delta Y / \Delta X$
the change in $Y$ per unit change in $X$

    $$
    \begin{array}{r}
    \ln \left(Y_{i t}\right)=Y_{b}-1 \\
    \ln \left(Y_{i t}\right)-\ln \left(Y_{i, t-1}\right)=\ln \left(\frac{Y_{i t}}{Y_{i, L-1}}\right) \\
    \frac{Y_{k,}-1}{Y_{i t-1}} \\
    \frac{Y_{i t}-Y_{i t-1}}{Y_{i, t-1}}
    \end{array}
    $$

    $\operatorname{Ln}(X)$ : The change in $Y$ (in units)related to a $1 \%$ increase in $X$
    $\operatorname{Ln}(Y)$ : The precent change in $Y$ related to a one-unit increate in $X$



### 格兰杰因果检验

检查X是否有助于Y的增长


### 时间序列分解

- Additive decomposition: $y_{t}=S_{t}+T_{t}+R_{t}$
- Multiplicative decomposition: $y_{t}=S_{t} \times T_{t} \times R_{t}$



### Spatial Panel data analysis
- Model specification could be a mixed structure of spatial lag and spatial error model.
- Unobserved heterogeneity could be fxed effects or random effects.
- OLS is biased and inconsistent; Consistent IV or 2SLS should be used, with robust inference.
- If normality assumption of the model is maintained, efficient ML estimation could be used but with computatlonal complexity.
- Efflclent GMM estimation is recommended.

- 案例：比如微信公众号流量影响微信视频号流量有溢出效应


### 各种相关系数

皮尔森相关系数 (Pearson) : 衡量了两个连续型变量之间的线性相关程度, 要求数据连续变量的取值服从正态分布
$$
\rho_{X, Y}=\frac{\operatorname{cov}(X, Y)}{\sigma_{X} \sigma_{Y}}=\frac{E\left[\left(X-\mu_{X}\right)\left(Y-\mu_{Y}\right)\right]}{\sigma_{X} \sigma_{Y}}
$$
斯皮尔曼相关系数（Spearman）：衡量两个变量之间秩次（排序的位置）的相关程度, 通常用于计算离散型数据、分类变量或等级变量
$$
\rho=\frac{\sum_{i}\left(x_{i}-\bar{x}\right)\left(y_{i}-\bar{y}\right)}{\sqrt{\sum_{i}\left(x_{i}-\bar{x}\right)^{2} \sum_{i}\left(y_{i}-\bar{y}\right)^{2}}} .
$$
之间的相关性, 对数据分布没有要求
肯德尔相关系数 (Kendall's Tau-b) : 用于计算有序的分类变量之间的相关系数, 和斯皮尔曼相关系数相似, 在样本较小时（比如小于12）更为精确
$$
\tau=\frac{(\text { number of concordant pairs })-(\text { number of discordant pairs })}{n(n-1) / 2} .
$$
