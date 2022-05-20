

- 定义：Inferring the effects of any treatment/policy/intervention/etc

## Motivation

辛普森悖论 Simpson's Paradox

定义：总体的规律跟分组的规律不同：几组不同的数据中均存在一种 趋势, 但当这些数据组合在一起后, 这种趋势消失或反转。

产生的原因：数据中存在多个变量。这些变量通常难以识别, 被称为“潜伏变量”。潜伏变量可能是由于采样错误造成的。

统计层面的核心：Partial correlation marginal correlationdramatically different

例子：

- [例1]: 男性用药多 但恢复率更高，所以让整体有这个correlation，不过by group的correlation不是这样！
<center>/images/CI_simpson_paradox.png" width="60%"/></center>

- [例2]: 比如说Y是购买概率，X是看直播时长，有可能实际上 男生女生都是看得越久 购买的概率越低，但由于普遍女生看的久，购买概率高，在整体数据上会体现出“观看直播时间越长，购买概率越高”的错误结论！

- [例3]: 给1500个人Treatment A 550个人B，看致死率，但本身有轻重症之分
<center>/images/CI_basic_1.png" width="75%"/></center>

现象：Subgroup之后结论是B更好、total的时候是A更好

造成的原因是unequal weighting：重症的人得到了更多的B导致B看起来不好了⇒总体的结论是一个有偏的加权平均

- Treatment A的致死率计算里，轻症的致死率权重大，所以致死率小
- Treatment B的致死率计算里，重症的致死率权重大，所以致死率大

正确的解释：Either treatment Atreatment B could be right answer, dependingcausal structure!

- 情景1: conditiona cause of the treatment 

<center>/images/CI_basic_2.png" width="40%"/><img src="../images/CI_basic_3. width="</center>

- 解释：比如医生把B专门用在重症，轻症的给A就可以。

- 选择：这种情况要选B（看sub-group的结论），因为我们B中的高致死率，只是因为他们本身就有很高的致死率被assign了更多的重症！

- 情景2: treatmenta cause of the condition 

<center>/images/CI_basic_4.png" width="35%"/><img src="../images/CI_basic_5. width="center>

- 解释：比如如果Treatment B，必须要等很久才能take，所以在等的时候condition会worsen——变成了重症！从而导致了更多的重症被接受了Treatment B

Treatment A不用等所以马上就轻症了！这就是为什么A的轻症占比更高

- 选择：这种情况要选A（看total的结论），因为Treatment本身是造成worse condition的原因，从而导致了Y。所以我们需要考虑T对C的影响而不能直接看conditional的结论（因为T会让你去不同的C！）

考虑这个影响的办法就是直接看T对Y的总体效果。

结论：conclusion取决于causal structure

Correlation  Imply Causation

- 例子：Cor——穿着鞋睡觉的人更容易头疼｜Cau——喝酒醉倒⇒穿着鞋睡觉、喝酒⇒头痛

喝酒是common cause

- 问题1——穿鞋睡的人和不穿鞋睡的人，除了穿鞋这一点之外还有喝酒这个重要的区别，所以不是控制其他变量不变
- 问题2——是喝酒同时导致了穿鞋睡觉和头痛。我们观察到可能主要是一个confounding effect （而不主要是Causal association的direct path）

 <center><img src="../images/CI_basic_6. width="center>

Total Association = mixture of causal and confounding association
> correlation是一种线性的association，不过也会跟general的混称

 does imply causation?

Inferring the effecttreatment/policy outcome的时候，怎样才能说明是causal呢？

 如果你知道做了某件事的potential outcome就是work，不做就没有，那就可以
<center>/images/CI_basic_7.png" width="65%"/></center>

- 问题：It is impossible to observe potential outcomes a given individual. —— CI的根本问题！

作为一个个人，如果你吃了药（吃药是factual），那你就不能知道你不吃药（不吃药是counterfactual）的outcome会是什么样了。

- 如果上升到群体然后平均的话，不是完全相等的！

Individual treatment effect )$

Average treatment effect (ATE):
$$
\begin{aligned}
\mathbb{E}\left[)\right]mathbb{E\mathbb{ \\
& \neq \mathbb{E]-\mathbb{E}
{aligned}
$$

- 第二个不等号左边是causal quantity，右边是measure of correlation，是不想等的因为有confounding effect（T除了直接对Y，还会通过C去影响Y）


 Randomized Control Trails （RCTs）

<center>/images/CI_basic_8.png" width="65%"/></center>

- 如果说你的treatment可以跟condition没关系的话，比如Treatment就是随机抛硬币跟C无关，那么就没有confounding effect（T跟Y不会通过C去影响）

- 然而这个要求我们的分组是相同的，比如穿鞋睡和不穿鞋睡是抛硬币随机的 不管你清醒与否！这样的效果就是清醒和喝酒的人，在穿不穿鞋这件事情上的分布都是一样的。

 observational studies

- 场景：Can’t always randomize treatment，比如given一些数据集

- 不道德｜Ethical reasons (eunethical to randomize people to smoke for measuring effect on lung cancer)
- 不可行｜Infeasibility . can’t randomize countries  communist/capitalist systems to measure effect on GDP)
- 不可能｜Impossibility . can’t change a living person’sbirth for measuring effect on breast cancer)

- 解决方案：adjust/control for confounders

## Potential Outcomes


上面提到的 无法观测what would have been if I也可以理解为一种缺失值处理问题：

$$
\begin{array}{cccccc}
\hline i\
\hline 1 \\





\hline
{array}
$$
- 因为individual只能有Y(treatment给1)或者Y(treatment给0)中的一个

Average treatment effect)

> 解决个体无法观测的方法

<center>/images/CI_basic_9.png" width="85%"/></center>

- ATE: $mathbb{E\mathbb{$
- association differencemathbb{E]-\mathbb{E}$

而这两个概念是不同的——correlation不是causation：Treatment之后的subgroup可能不是在所有其他变量上comparable的：
<center>/images/CI_basic_10. width="center>


什么时候可以呢？

- Ignorability—— T$

$$
\begin{aligned}
\mathbb{-\mathbb\mathbb{]-\mathbb{E}) \mid T\quad \text { (ignorability)\
&=\mathbb{E}-\mathbbY \mid T=0]
{aligned}
$$

- 概率图的解释：因为confounding variable跟T无关，也就没有了confounding effect!

<center>/images/CI_basic_11. width="center>

- Exchangeability: 两组可以交换但期望不变，是上一个独立条件的另一种解释

<center>/images/CI_basic_12. width="center>

Ignorability和Exchangeability假设在成立的的时候，能给我们identifiability of causal effect
identifiableA causal quantity (e$\mathbbis identifiable if we can compute it from a purely statistical quantity mathbb{E )

然而让他们成立，需要RCT来让被测完全随机地执行treatment：

 <center><img src="../images/CI_basic_13" width=/center>

- 结果：让最后的组完全comparable

- 概率图：

<center>/images/CI_basic_14. width="center>

Conditional exchangeability假设

很多时候假设exchangeable不可行，但我们可以control for relevant variables by conditioning

`Conditional exchangeability` (Unconfoundedness / conditional ignorability):X$

<center>/images/CI_basic_15. width="center>

- 含义：在相同的X level中就没有X的confounding了，因为X被控制了而同一个X中的treatment groups是comparable的。从而X确定时 T和Y就没有非causal的confounding effect了

- 存在的问题：这是一个untestable的assumption，因为可能有另一个unobserved的confounder W:

<center>/images/CI_basic_16. width="center>

`Conditional`:

$$\begin{aligned} \mathbb{E}=\mathbb \mid X]-\mathbb \mid X]&=\mathbb{E}) \mid Tmathbb{E=\mathbbY \mid Tmathbb{E \end{aligned}$$

- 第一行：linearity of expectation
- 第二行：conditional exchangeability因为X相同无confounding
- 第三行：得出的结果就是两个statistical quantities

从Conditional ATE也可以推到marginal effect:

$$\begin{aligned} \mathbb{E}&=\mathbb{E} \mathbb) \mid Xmathbb{E}[\mathbb{E}\mathbb{ \mid T=aligned}$$

Positivity假设
 values of covariates X present in the population of interest: $
 
 这个保证了上面的式子不会遇到除0的问题，具体可以见课本P12

 Intuition: 如果总体中有一个x都没有接收到treatment，我们不知道他们如果接受treatment的话会怎么样


 <center><img src="../images/CI_basic_17" width=/center>



 Overlap
另一个解读Positivity的视角：`overlap` \mid T=$ and $\ \mid T=1)$

我们希望covariate distribution of the treatment group to overlap with the covariate distributioncontrol group!

<center>/images/CI_basic_18. width="center>



所以overlap也叫common support

 Positivity-Unconfoundedness Tradeoff

这里Positivity和unconfoundedness是一个tradeoff，如果我们增加变量，那么subgroup就会变小，虽然更容易解决confound，但是会让一些group整个T/T的概率增高（比如subgroup的size缩小到只有1）


 Extrapolation

违反Positivity的后果是影响模型的表现，因为通常causal effect需要根据(t,y)的样本拟合$\mathbb{ \mid t,

<center>/images/CI_basic_19. width="center>

 在实际操作中获得$\mathbb{E}的方法是用任何一个minimize MSE的model就可以（比如Linear Regression）

No Interference假设

My outcome is unaffectedanyone else’s treatment. Rather,outcome is only a function of my treatment.

\ldots,  \ldots,n}\righti}\right)$$

<center>/images/CI_basic_20. width="center>


Consistency假设

outcome we observe Yactually potential outcome under observed treatment.

If the treatment then the observed outcome $Y$ is the potential outcome under treatment  Formally,
$$
\Longrightarrow t)
$$
We could write this equivalentlyfollow:
$$
T)
$$

<center>/images/CI_basic_21. width="center>

总结

之前的conditional ATE⇒marginal ATE其实依赖了以上的四个假设
<center>/images/CI_basic_22. width="center>

 interference: 保证我们只用关注$\mathbb{]$来衡量causal effect
- unconfoundedness: 保证了T能加在X上面
- positivity：保证了每个X都有T所以能evaluate
- consistency：保证T了之后的Y就是我们想要的


## The Flow of Association and CausationGraphs

### Bayesian Networks



### Causal Graphs



### Graphs Graphical Building Blocks



### Chains and Forks



### Colliders and their Descen-dants 



### d-separation 



### Association  sation


