# 基础

- 定义：Inferring the effects of any treatment/policy/intervention/etc

## Motivation

### 辛普森悖论 Simpson's Paradox

定义：总体的规律跟分组的规律不同：几组不同的数据中均存在一种 趋势, 但当这些数据组合在一起后, 这种趋势消失或反转。

产生的原因：数据中存在多个变量。这些变量通常难以识别, 被称为“潜伏变量”。潜伏变量可能是由于采样错误造成的。

统计层面的核心：Partial correlation and marginal correlation can be dramatically different

例子：

- [例1]: 男性用药多 但恢复率更高，所以让整体有这个correlation，不过by group的correlation不是这样！
    <center><img src="../images/CI_simpson_paradox.png" width="60%"/></center>

- [例2]: 比如说Y是购买概率，X是看直播时长，有可能实际上 男生女生都是看得越久 购买的概率越低，但由于普遍女生看的久，购买概率高，在整体数据上会体现出“观看直播时间越长，购买概率越高”的错误结论！

- [例3]: 给1500个人Treatment A 550个人B，看致死率，但本身有轻重症之分
    <center><img src="../images/CI_basic_1.png" width="75%"/></center>

    现象：Subgroup之后结论是B更好、total的时候是A更好

    造成的原因是unequal weighting：重症的人得到了更多的B导致B看起来不好了⇒总体的结论是一个有偏的加权平均
    
    - Treatment A的致死率计算里，轻症的致死率权重大，所以致死率小
    - Treatment B的致死率计算里，重症的致死率权重大，所以致死率大

    正确的解释：Either treatment A or treatment B could be the right answer, depending on the causal structure of the data!

    - 情景1: condition is a cause of the treatment 

        <center><img src="../images/CI_basic_2.png" width="40%"/><img src="../images/CI_basic_3.png" width="33.5%"/></center>

        - 解释：比如医生把B专门用在重症，轻症的给A就可以。

        - 选择：这种情况要选B（看sub-group的结论），因为我们B中的高致死率，只是因为他们本身就有很高的致死率被assign了更多的重症！

    - 情景2: treatment is a cause of the condition 

        <center><img src="../images/CI_basic_4.png" width="35%"/><img src="../images/CI_basic_5.png" width="40%"/></center>

        - 解释：比如如果Treatment B，必须要等很久才能take，所以在等的时候condition会worsen——变成了重症！从而导致了更多的重症被接受了Treatment B

            Treatment A不用等所以马上就轻症了！这就是为什么A的轻症占比更高

        - 选择：这种情况要选A（看total的结论），因为Treatment本身是造成worse condition的原因，从而导致了Y。所以我们需要考虑T对C的影响而不能直接看conditional的结论（因为T会让你去不同的C！）
        
            考虑这个影响的办法就是直接看T对Y的总体效果。

    结论：conclusion取决于causal structure

### Correlation Does Not Imply Causation

- 例子：Cor——穿着鞋睡觉的人更容易头疼｜Cau——喝酒醉倒⇒穿着鞋睡觉、喝酒⇒头痛

    喝酒是common cause

    - 问题1——穿鞋睡的人和不穿鞋睡的人，除了穿鞋这一点之外还有喝酒这个重要的区别，所以不是控制其他变量不变
    - 问题2——是喝酒同时导致了穿鞋睡觉和头痛。我们观察到可能主要是一个confounding effect （而不主要是Causal association的direct path）

         <center><img src="../images/CI_basic_6.png" width="35%"/></center>

        Total Association = mixture of causal and confounding association
        > correlation是一种线性的association，不过也会跟general的混称

### what does imply causation?

Inferring the effect of treatment/policy on some outcome的时候，怎样才能说明是causal呢？

方法1: 如果你知道做了某件事的potential outcome就是work，不做就没有，那就可以
<center><img src="../images/CI_basic_7.png" width="65%"/></center>

- 问题：It is impossible to observe all potential outcomes for a given individual. —— CI的根本问题！

    作为一个个人，如果你吃了药（吃药是factual），那你就不能知道你不吃药（不吃药是counterfactual）的outcome会是什么样了。

- 如果上升到群体然后平均的话，不是完全相等的！

    Individual treatment effect (ITE): $Y_{i}(1)-Y_{i}(0)$
    
    Average treatment effect (ATE):
    $$
    \begin{aligned}
    \mathbb{E}\left[Y_{i}(1)-Y_{i}(0)\right] &=\mathbb{E}[Y(1)]-\mathbb{E}[Y(0)] \\
    & \neq \mathbb{E}[Y \mid T=1]-\mathbb{E}[Y \mid T=0]
    \end{aligned}
    $$

    - 第二个不等号左边是causal quantity，右边是measure of correlation，是不想等的因为有confounding effect（T除了直接对Y，还会通过C去影响Y）


方法2: Randomized Control Trails （RCTs）

<center><img src="../images/CI_basic_8.png" width="65%"/></center>

- 如果说你的treatment可以跟condition没关系的话，比如Treatment就是随机抛硬币跟C无关，那么就没有confounding effect（T跟Y不会通过C去影响）

- 然而这个要求我们的分组是相同的，比如穿鞋睡和不穿鞋睡是抛硬币随机的 不管你清醒与否！这样的效果就是清醒和喝酒的人，在穿不穿鞋这件事情上的分布都是一样的。

方法3: observational studies

- 场景：Can’t always randomize treatment，比如given一些数据集

    - 不道德｜Ethical reasons (e.g. unethical to randomize people to smoke for measuring effect on lung cancer)
    - 不可行｜Infeasibility (e.g. can’t randomize countries into communist/capitalist systems to measure effect on GDP)
    - 不可能｜Impossibility (e.g. can’t change a living person’s DNA at birth for measuring effect on breast cancer)

- 解决方案：adjust/control for confounders

## Potential Outcomes


上面提到的 无法观测what would have been if I也可以理解为一种缺失值处理问题：

$$
\begin{array}{cccccc}
\hline i & T & Y & Y(1) & Y(0) & Y(1)-Y(0) \\
\hline 1 & 0 & 0 & ? & 0 & ? \\
2 & 1 & 1 & 1 & ? & ? \\
3 & 1 & 0 & 0 & ? & ? \\
4 & 0 & 0 & ? & 0 & ? \\
5 & 0 & 1 & ? & 1 & ? \\
6 & 1 & 1 & 1 & ? & ? \\
\hline
\end{array}
$$
- 因为individual只能有Y(treatment给1)或者Y(treatment给0)中的一个

### Average treatment effect(ATE)

> 解决个体无法观测的方法

<center><img src="../images/CI_basic_9.png" width="85%"/></center>

- ATE: ${E}[Y(1)-Y(0)]=\mathbb{E}[Y(1)]-\mathbb{E}[Y(0)]$
- association difference: $\mathbb{E}[Y \mid T=1]-\mathbb{E}[Y \mid T=0]$

而这两个概念是不同的——correlation不是causation：Treatment之后的subgroup可能不是在所有其他变量上comparable的：
<center><img src="../images/CI_basic_10.png" width="85%"/></center>


什么时候可以呢？

- Ignorability—— $(Y(1), Y(0)) \perp T$

    $$
    \begin{aligned}
    \mathbb{E}[Y(1)]-\mathbb{E}[Y(0)] &=\mathbb{E}[Y(1) \mid T=1]-\mathbb{E}[Y(0) \mid T=0] \quad \text { (ignorability) } \\
    &=\mathbb{E}[Y \mid T=1]-\mathbb{E}[Y \mid T=0]
    \end{aligned}
    $$

    - 概率图的解释：因为confounding variable跟T无关，也就没有了confounding effect!

        <center><img src="../images/CI_basic_11.png" width="85%"/></center>

- Exchangeability: 两组可以交换但期望不变，是上一个独立条件的另一种解释

    <center><img src="../images/CI_basic_12.png" width="85%"/></center>

Ignorability和Exchangeability假设在成立的的时候，能给我们identifiability of causal effect
- **identifiable**: A causal quantity (e.g. $\mathbb{E}[Y(t)])$ is identifiable if we can compute it from a purely statistical quantity (e.g. $\mathbb{E}[Y \mid t]$ )

然而让他们成立，需要RCT来让被测完全随机地执行treatment：

 <center><img src="../images/CI_basic_13.png" width="75%"/></center>

- 结果：让最后的组完全comparable

- 概率图：

    <center><img src="../images/CI_basic_14.png" width="85%"/></center>

### Conditional exchangeability假设

很多时候假设exchangeable不可行，但我们可以control for relevant variables by conditioning

`Conditional exchangeability` (Unconfoundedness / conditional ignorability): $(Y(1), Y(0)) \perp T \mid X$

<center><img src="../images/CI_basic_15.png" width="85%"/></center>

- 含义：在相同的X level中就没有X的confounding了，因为X被控制了而同一个X中的treatment groups是comparable的。从而X确定时 T和Y就没有非causal的confounding effect了

- 存在的问题：这是一个untestable的assumption，因为可能有另一个unobserved的confounder W:

    <center><img src="../images/CI_basic_16.png" width="85%"/></center>

`Conditional ATE`:

$$\begin{aligned} \mathbb{E}[Y(1)-Y(0) \mid X] &=\mathbb{E}[Y(1) \mid X]-\mathbb{E}[Y(0) \mid X] \\ &=\mathbb{E}[Y(1) \mid T=1, X]-\mathbb{E}[Y(0) \mid T=0, X] \\ &=\mathbb{E}[Y \mid T=1, X]-\mathbb{E}[Y \mid T=0, X] \end{aligned}$$

- 第一行：linearity of expectation
- 第二行：conditional exchangeability因为X相同无confounding
- 第三行：得出的结果就是两个statistical quantities

从Conditional ATE也可以推到marginal effect:

$$\begin{aligned} \mathbb{E}[Y(1)-Y(0)] &=\mathbb{E}_{X} \mathbb{E}[Y(1)-Y(0) \mid X] \\ &=\mathbb{E}_{X}[\mathbb{E}[Y \mid T=1, X]-\mathbb{E}[Y \mid T=0, X]] \end{aligned}$$

### Positivity假设
 For all values of covariates X present in the population of interest: $0<P(T=1 \mid X=x)<1$
 
 这个保证了上面的式子不会遇到除0的问题，具体可以见课本P12

 Intuition: 如果总体中有一个x都没有接收到treatment，我们不知道他们如果接受treatment的话会怎么样


 <center><img src="../images/CI_basic_17.png" width="45%"/></center>



#### Overlap
另一个解读Positivity的视角：`overlap` of $P(X \mid T=0) \quad$ and $\quad P(X \mid T=1)$

我们希望covariate distribution of the treatment group to overlap with the covariate distribution of the control group!

<center><img src="../images/CI_basic_18.png" width="45%"/></center>



所以overlap也叫common support

#### Positivity-Unconfoundedness Tradeoff

这里Positivity和unconfoundedness是一个tradeoff，如果我们增加变量，那么subgroup就会变小，虽然更容易解决confound，但是会让一些group整个T/not T的概率增高（比如subgroup的size缩小到只有1）


#### Extrapolation

违反Positivity的后果是影响模型的表现，因为通常causal effect需要根据(t, x, y)的样本拟合$\mathbb{E}[Y \mid t, x]$

<center><img src="../images/CI_basic_19.png" width="65%"/></center>

- 注意 在实际操作中获得$\mathbb{E}[Y \mid T=1, x]$的方法是用任何一个minimize MSE的model就可以（比如Linear Regression）

### No Interference假设

My outcome is unaffected by anyone else’s treatment. Rather, my outcome is only a function of my own treatment.

$$Y_{i}\left(t_{1}, \ldots, t_{i-1}, t_{i}, t_{i+1}, \ldots, t_{n}\right)=Y_{i}\left(t_{i}\right)$$

<center><img src="../images/CI_basic_20.png" width="65%"/></center>


### Consistency假设

The outcome we observe Y is actually the potential outcome under the observed treatment.

If the treatment is $T$, then the observed outcome $Y$ is the potential outcome under treatment $T$. Formally,
$$
T=t \Longrightarrow Y=Y(t)
$$
We could write this equivalently as follow:
$$
Y=Y(T)
$$

<center><img src="../images/CI_basic_21.png" width="65%"/></center>

### 总结

之前的conditional ATE⇒marginal ATE其实依赖了以上的四个假设
<center><img src="../images/CI_basic_22.png" width="65%"/></center>

- no interference: 保证我们只用关注$\mathbb{E}[Y(1)-Y(0)]$来衡量causal effect
- unconfoundedness: 保证了T能加在X上面
- positivity：保证了每个X都有T所以能evaluate
- consistency：保证T了之后的Y就是我们想要的


## The Flow of Association and Causation Graphs




### Causal Graphs

- `cause`: $A$ variable $X$ is said to be a cause of a variable $Y$ if $Y$ can change in response to changes in $X$

- `Strict Causal Edges Assumption`:  In a directed graph, every parent is a direct cause of all its children.

- 2 main assumptions that we need for our causal graphical models:

    <center><img src="../images/CI_basic_23.png" width="65%"/></center>

### Two-Node Graphs and Graphical Building Blocks

Building blocks:

 <center><img src="../images/CI_basic_24.png" width="65%"/></center>

- `Chains` / `Forks`: 有common cause 所以X1和X3也是association，只要common cause改变两个会联动改变，所以有association flow

    - 然而如果conditional on X2的话相当于block了这个path：

     <center><img src="../images/CI_basic_25.png" width="65%"/></center>

    - 通过Bayes network factorization和Bayes Rule证明可以得到：$P\left(x_{1}, x_{3} \mid x_{2}\right) =\frac{P\left(x_{1}, x_{2}\right)}{P\left(x_{2}\right)} P\left(x_{3} \mid x_{2}\right)$

- `Immorality`: 两个没connect的parents有一个共同的child

    这个child不允许association flow through it而是直接blcok了，管它叫`collider`

    - 这个图中本身$P\left(x_{1}, x_{3}\right) = P(x_{1})P(x_{3})$

    - 但当我们conditional on X2之后就会unblock这个path！

        例子：

         <center><img src="../images/CI_basic_26.png" width="65%"/></center>

         <center><img src="../images/CI_basic_27.png" width="85%"/></center>

         这个其实就是selection bias！



### d-separation 

`Blocked path`: A path between nodes $X$ and $Y$ is blocked by a (potentially empty) conditioning set $\mathrm{Z}$ if either of the following is true:
1. Along the path, there is a chain $\cdots \rightarrow W \rightarrow \cdots$ or a fork $\cdots \leftarrow W \rightarrow \cdots$ where $W$ is conditioned on $(W \in Z)$.
2. There is a collider $W$ on the path that is not conditioned on $(W \notin Z)$ and none of its descendants are conditioned on $(\operatorname{de}(W) \nsubseteq Z Z)$.

`Unblocked path`: a path that is not blocked

`d-seperation`: Two (sets of) nodes $\mathrm{X}$ and $\mathrm{Y}$ are d-separated by a set of nodes $\mathrm{Z}$ if all of the paths between (any node in) $X$ and (any node in) $Y$ are blocked by $Z$.

- `Global Markov Assumption`: Given that $P$ is Markov with respect to $G$ (satisfies the local Markov assumption), if $X$ and $Y$ are $d$-separated in $G$ conditioned on $Z$, then $X$ and $Y$ are independent in $P$ conditioned on $Z$. We can write this succinctly as follows:
    $$
    X \perp_{G} Y\left|Z \Longrightarrow X \perp_{P} Y\right| Z
    $$

    会得到conditional indolence

总结：
- association flows along chains and forks, unless a node is conditioned on
- a collider blocks the flow of association, unless it is conditioned on


### Flow of Association and Causation


d-separation Implies Association is Causation!

<center><img src="../images/CI_basic_28.png" width="45%"/></center>

- 原因：通过这个我们确保non-causal association不会flow，剩下的就只有causation了
- 记得：association is not causation！


## Causal Models

我们需要Causal Model来将因果被估量，变成统计被估量
<center><img src="../images/CI_basic_29.png" width="45%"/></center>


### The $do$-operator

用`do`来表示intervention:

- Conditioning只是restrict to subset of data, 而intervene是对整体数据问what would be like if XXX
<center><img src="../images/CI_basic_30.png" width="45%"/></center>

- ATE从而可以写成：$\mathbb{E}[Y \mid d o(T=1)]-\mathbb{E}[Y \mid d o(T=0)]$

- 可以观测到的：$P(Y, T, X)$、$P(Y \mid T=t)$
    
    - 没有do所以可以直接observe data from them without needing to carry out any experiment

    Interventional的：$P(Y \mid d o(T=t))$、$P(Y \mid d o(T=t), X=x)$

    - Condition on do的含义：everything in that expression is in the post-intervention world where the intervention do(t) occurs

- Indentification就是要把Interventional的变成observational的 （remove do!)，但是不一直能work

    <center><img src="../images/CI_basic_31.png" width="45%"/></center>

### Main assumption: modularity

- `Causal Mechanism`:  causal mechanism that generates $X_{i}$ is all of $X_{i}$ 's parents and their edges that go into $X_{i}$

    <center><img src="../images/CI_basic_32.png" width="45%"/></center>


- `Modularity / Independent Mechanisms / Invariance`: If we intervene on a set of nodes $S \subseteq[n]$($\{1,2, \ldots, n\}$) setting them to constants, then for all $i$, we have the following:
    1. If $i \notin S$, then $P\left(x_{i} \mid \mathrm{pa}_{i}\right)$ remains unchanged.
    2. If $i \in S$, then $P\left(x_{i} \mid \mathrm{pa}_{i}\right)=1$ if $x_{i}$ is the value that $X_{i}$ was set to by the intervention; otherwise
    
    - Intuition: If we intervene on a node $\mathrm{X}_{\mathrm{i}}$, then only the mechanism $P\left(x_{i} \mid \mathrm{pa}_{i}\right)$ changes. All other mechanisms $P\left(x_{j} \mid \mathrm{pa}_{j}\right)$ where $i \neq j$ remain unchanged
    
        In other words, the causal mechanisms are modular.

- 用途：Truncated Factorization

    $
    P\left(x_{1}, \ldots, x_{n} \mid d o(S=s)\right)=\prod_{i \notin S} P\left(x_{i} \mid \mathrm{pa}_{i}\right)
    $
    
    Otherwise, $P\left(x_{1}, \ldots, x_{n} \mid d o(S=s)\right)=0$

    The latter's product is only over $i \notin S$ rather than all $i$.

### Backdoor adjustment


### Structural causal models


### A complete example with estimation


