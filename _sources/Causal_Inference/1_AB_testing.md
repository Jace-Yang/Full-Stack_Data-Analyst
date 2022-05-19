# AB Testing


定义：AB testing (controlled experiments) is a type of experiment with two variants, A and B, which are the control and variation in a controlled experiment.

- 一般Control是A——existing features，Treatment是B——new features 
- 会对部分用户展开


定位：
- 纵向：数据分析、用户调研是理解需求的方法，ABtest是验证方案的工具，是互相补充的上下游的关系！
- 横向：用户访谈、问卷调查、焦点组也是evaluate方案的方法，AB实验会给broad quantitive data，上面的会给你deep qualitative data，也可以形成组合拳！



## ABTest的设计

ABtest其实就是控制变量法。为了评估测试和验证模型/项目的效果，在app/pc端设计出多个版本，在同一时间维度下，分别用组成相同/相似的群组去随机访问这些版本，记录下群组的用户体验数据和业务数据，最后评估不同方案的效果决定是否上线。

- Time——运行多久
- Size——决定样本量

    - Type Il error or power
    - Significance level
    - Minimum detectable effect: smallest different matters in practice

    $$\text { sample size } \simeq \frac{16 \sigma^{2}}{\delta^{2}}$$

    - $$\delta \text { : difference between treatment and control }$$

步骤：

```python
1. 基于现状和期望，分析并提出假设
2. 设定指标目标（核心指标）
3. 设计与开发
4. 分配流量进行测试
5. 埋点采集数据
6. 实验后分析数据
7. 发布新版本/改进设计方案/调整流量继续测试
```




## ABtest背后的理论支撑是什么？

**中心极限定理**：在样本量足够大的时候，可以认为样本的均值近似服从正态分布。

**假设检验**：假设检验是研究如何根据抽样后获得的样本来检查抽样前所作假设是否合理，**A/B Test 从本质上来说是一个基于统计的假设检验过程**，它首先对实验组和对照组的关系提出了某种假设，然后计算这两组数据的差异和确定该差异是否存在统计上的显著性，最后根据上述结果对假设做出判断。

假设检验的核心是**证伪**，所以原假设是统计者想要拒绝的假设，无显著差异我们也可以理解为：实验组和对照组的统计差异是由抽样误差引起的（误差服从正态分布）。

## 如何分组才能更好地避免混淆呢？

AB实验需要利用控制变量法的原理, 确保A、B两个方案中只有一个不同的变量, 其他变量保持一致，

- 必须满足的条件
    - 特征相同 (或相似) 的用户群组
    - 同一时间维度
- 操作方法：
    - 利用用户的唯一标识的尾号或者其他标识进行分类，如按照尾号的奇数或者偶数将其分为两组。
    - 用一个hash函数将用户的唯一标识进行hash取模，分桶。可以将用户均匀地分到若干个桶中，如分到100个或者1000个桶中，这样的好处就是可以进一步将用户打散，提高分组的效果。

当然，如果有多个分组并行进行的情况的话，要考虑独占域和分享域问题。（不同域之间的用户相互独立，交集为空）对于共享域，我们要进行分层。但是在分层中，下一层要将上一层的用户打散，确保下一层用户的随机性。



## 样本量大小如何？

​	理论上，我们想要样本量越多的越好，因为这样可以避免第二类错误。随着样本量增加，power=1-β也在增大，一般到80%，这里我们可以算出一个最小样本量，但理论上样本量还是越大越好。

实际上，样本量越少越好，这是因为

1. 流量有限：小公司就这么点流量，还要精打细算做各种测试，开发各种产品。在保证样本分组不重叠的基础上，产品开发速度会大大降低。

2. 试错成本大：如果拿50%的用户做实验，一周以后发现总收入下降了20%，这样一周时间的实验给公司造成了10%的损失，这样损失未免有点大。

## 两类错误是什么？

1. **弃真**：实验组和对照组没有显著差异，但我们接受了方案推了全量。减少这种错误的方法就是提高显著性水平，比如 p 值小于 0.05 才算显著，而不是小于 0.1，显著性水平是人为给定的犯一类错误的可以接受的上限（$p$值为犯 I 类错误的概率$\alpha$ ）。

2. **存伪**：实验组和对照组有显著差异，但我们没有接受方案。

   II 类错误和**统计功效 (power)** 有关，统计功效可以简单理解为真理能被发现的可能性。统计功效 为:$1-\beta$ ，而$\beta$为犯第二类错误的概率。影响统计功效的因素有很多，主要的有三个：统计量、样本量和 I 类错误的概率$\alpha$  。

## 埋点&暗中观察

​	当我们确定了需要分析的具体指标之后，就需要我们进行埋点设计，把相关的用户行为收集起来，供后续的流程进行数据分析，从而得出实验结论。

​	对于 ABTest我们需要知道当前用户是处于对照组还是实验组，所以埋点中这些参数必须要有。埋点完了就是收集实验数据了（暗中观察），主要看以下两个方面：

1. 观察样本量是否符合预期，比如实验组和对照组分流的流量是否均匀，正常情况下，分流的数据不会相差太大，如果相差太大，就要分析哪里出现了问题。

2. 观察用户的行为埋点是否埋的正确，很多次实验之后，我们发现埋点埋错了。




## 注意点

### Novelty and Primacy Effect

在新进行实验版本变更的短期内会有两种效应，但不会一直持续：
- Primacy effect: People are reluctant to change
- Novelty effect: People welcome the changes and use more

这两个效应会导致不同的initial effect

```
Sample question:
- Ran an A/B test on a new feature
- The test won and we launched the change
- After a week, the treatment effect quickly declined

Answer:
- Novelty effect
- Repeat usage declined when effect wears off
```

解决方案：

- Run tests only on first time users: 这样不会被Novelty和Primacy影响

- 如果已经有在跑的实验，想看这些effect的影响的话

### INTERFERENCE BETWEEN VARIANTS

Network effect: 控制组的用户会被实验组的用户影响
- User behaviors are impacted by others
- The effect can spillover the control group


Two-sided markets: 实验组和控制组会竞争一样的资源
- Eg. Uber, Lyft, Airbnb
- Resources are shared among control and treatment groups
- Eg. treatment group attracts more drivers
- less drivers are available for control group

解决方案

Sample questions:
- A new feature provides coupons to our riders
- Goal: increase rides by decreasing price
- Testing strategy: evaluate the effect of the new feature

防止用户互相影响的方案——Isolate Users

- 两端用户场景

    Geo-based randomization: Split by geolocations
    - Eg. New York vs. San Francisco
    - Big variance since markets are unique

    Time-based randomization: Split by day of week
    - Assign all users to either treatment or control
    - Only when treatment effect is in short time
    - 只有在effect短（比如uber打车）的时候有用，但长时间的（比如是否推荐其他用户）就没用

- Network场景

    Create network clusters:
    - People interact mostly within the cluster
    - Assign clusters randomly

    Ego-network randomization: 
    - Originated from LinkedIn
    - An ego network cluster is defined as a portion of a social network formed of a given individual, termed ego, and the other persons with whom she has a social relationship, termed alters
    - One-out network effect: user either has the feature or not
    - It's simpler and more scalable
        


### 一人多账号

如果一个人有多个账号，分别做不同用途，abtest的时候怎么分组才最合理呢？：我们对这类人的分类是，看的不是他是谁，而是他做了什么。按照我们对行业的分类，行为不同的话就是两类人，和身份证是不是同一个无关。我们要聚合的是有相同行为特征的账户，而不是人。



## 结果分析

### Multiple Testing Problem
如果有三个组，有一个组p-value <0.05，也不能执行，因为此时

- $\operatorname{Pr}($ no false positive $)=(1-0.05)^{3}=0.95^{3}=0.857$
- $\operatorname{Pr}($ at least 1 false positive $)=1-\operatorname{Pr}$ (no false positive) = 0.143
- Type I error over 14\%

解决方案1：Bonferroni  correction
- Significance level / number of tests
    - 比如：Significance level 10 tests $=0.05 / 10=0.005$

- 缺点：太过保守


 解决方案2：False Discovery Rate

$\mathrm{FDR}=E\left[\frac{\text { false positives }}{\text { rejections }}\right]$



## 参考资料

https://zhuanlan.zhihu.com/p/165406531


