# 基础

- 定义：Inferring the effects of any treatment/policy/intervention/etc

## Motivation

### 辛普森悖论 Simpson's Paradox

总体的规律跟分组的规律不同

统计层面的核心是：Partial correlation and marginal correlation can be dramatically different

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

- 然而 作为一个个人，如果你吃了药（吃药是factual），那你就不能知道你不吃药（不吃药是counterfactual）的outcome会是什么样了。

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

