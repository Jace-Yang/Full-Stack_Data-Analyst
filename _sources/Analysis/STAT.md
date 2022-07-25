# 统计分析



### 参数估计

https://blog.csdn.net/liuyuemaicha/article/details/52497512

- 用样本统计量去估计总体的参数。

- 分类

  - 点估计：依据样本估计总体分布中所含的未知参数或未知参数的函数。

  - 区间估计（置信区间的估计）：依据抽取的样本，根据一定的正确度与精确度的要求，构造出适当的区间，作为总体分布的未知参数或参数的函数的真值所在范围的估计。例如人们常说的有百分之多少的把握保证某值在某个范围内，即是区间估计的最简单的应用。

- 估计方法

  - 矩估计法：

    - 矩估计法的理论依据是大数定律，主要基于用样本矩估计总体矩的思想

    - 矩的理解：在数理统计学中一类**数字特征**称为矩。

  - 极大似然估计总结
    - 似然函数直接求导一般不太好求,一般得到似然函数L(θ)之后,都是先求它的对数,即ln L(θ),因为ln函数不会改变L的单调性
    - 对ln L(θ)求θ的导数,令这个导数等于0,得到驻点.在这一点,似然函数取到最大值,所以叫最大似然估计法
    - 本质原理：似然估计是已知结果去求未知参数，对于已经发生的结果（一般是一系列的样本值）既然他会发生说明在未知参数θ的条件下，这个结果发生的可能性很大。
      - 所以最大似然估计求的就是使这个结果发生的可能性最大的那个θ，这个有点后验的意思

- 置信度、置信区间

  - 置信区间是我们所计算出的变量存在的范围，置信水平就是我们对于这个数值存在于我们计算出的这个范围的可信程度。

    - 举例来讲，有95%的把握，真正的数值在我们所计算的范围里。

    - 在这里，95%是置信水平，而计算出的范围，就是置信区间。

    - 如果置信度为95%， 则抽取100个样本来估计总体的均值，由100个样本所构造的100个区间中，约有95个区间包含总体均值。



### 假设检验

**跟参数估计的区别和联系：**参数估计和假设检验是统计推断的两个组成部分，都是利用样本对总体进行某种推断，但推断的角度不同。

- 参数估计讨论的是用样本估计总体参数的方法，总体参数μ在估计前是未知的。
- 在假设检验中，则是先对μ的值提出一个假设，然后利用样本信息去检验这个假设是否成立。

常用检验形式：

- T检验
  - 场景：不知道总体标准差需要用样本的标准差来代替、适用于小样本
- Z检验
  - 场景：知道总体的标准差、适用于大样本

p值：H0成立下出现这个统计量及它更极端情况的概率

- 例：双侧检验硬币公平得到8次正面：把八次正面的概率，与更极端的九次正面、十次正面的概率，以及012的加起来，在0.5的情况下只有0.1，这个就是p值



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
