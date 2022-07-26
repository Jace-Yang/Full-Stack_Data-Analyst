# 数据分析

常见Tradeoff：最大化哪一个：
- 收益ROI = LT × ARPU / CPA
- 规模DAU = DNU × LT + RDAU



## 各赛道指标体系

这个模块是适合用思维导图整理的part！用飞书来整理啦：[指标体系](https://x11awkb08o.feishu.cn/docs/doccn1BQjkvMv2QOezX6DLngx1e)



## 分析方法

### 指标纵向下钻

可以通过人群异质性效果探查（HTE），找到策略未触达的细分人群

<center><img src="../images/DA_metrics_1.png" width="35%"/></center>

### 指标横向分层

- 基于核心KPI指标：比如低转化率人群、中等、高转化率人群，知道运营动作要对的人
- 基于人群特征：年龄职业等，其实这些对购买习惯也有影响
- 基于场景：比如不同渠道、界面

RFM——对于这些分层方法的一种交叉



### Case

**如果次日用户留存率下降了 5%该怎么分析？**

- 首先采用“两层模型”分析：对用户进行细分，包括新老、渠道、活动、画像等多个维度，然后分别计算每个维度下不同用户的次日留存率。通过这种方法定位到导致留存率下降的用户群体是谁。

- 对于目标群体次日留存下降问题，具体情况具体分析。具体分析可以采用“内部-外部”因素考虑。

  - a. 内部因素找链路——分为获客（渠道质量低、活动获取非目标用户）、满足需求（新功能改动引发某类用户不满）、提活手段（签到等提活手段没达成目标、产品自然使用周期低导致上次获得的大量用户短期内不需要再使用等）；

  - b. 外部因素采用PEST分析（宏观经济环境分析），政治（政策影响）、经济（短期内主要是竞争环境，如对竞争对手的活动）、社会（舆论压力、用户生活方式变化、消费心理变化、价值观变化等偏好变化）、技术（创新解决方案的出现、分销渠道变化等）。

**一个网站销售额变低，你从哪几个方面去考量？**

- 首先要定位到现象真正发生的位置，到底是谁的销售额变低了？这里划分的维度有：

  - a. 用户（画像、来源地区、新老、渠道等）

  - b. 产品或栏目

  - c. 访问时段

- 定位到发生位置后，进行问题拆解，关注目标群体中哪个指标下降导致网站销售额下降：

  - a. 销售额=入站流量x下单率x客单价

  - b. 入站流量 = Σ各来源流量x转化率

  - c. 下单率 = 页面访问量x转化率

  - d. 客单价 = 商品数量x商品价格

- 确定问题源头后，对问题原因进行分析，如采用内外部框架：

  - a. 内部：网站改版、产品更新、广告投放

  - b. 外部：用户偏好变化、媒体新闻、经济坏境、竞品行为等

**用户流失的分析**

- 两层模型：

  - 细分用户、产品、渠道，看到底是哪里用户流失了。注意由于是用户流失问题，所以这里细分用户时可以细分用户处在生命周期的哪个阶段。
  - 指标拆解：用户流失数量 = 该群体用户数量*流失率。拆解，看是因为到了这个阶段的用户数量多了（比如说大部分用户到了衰退期），还是这个用户群体的流失率比较高

- 内外部分析：

  - a. 内部：新手上手难度大、收费不合理、产品服务出现重大问题、活动质量低、缺少留存手段、用户参与度低等

  - b. 外部：市场、竞争对手、社会环境、节假日等

- **（2）新用户流失和老用户流失有什么不同：**

  - 新用户流失：原因可能有非目标用户（刚性流失）、产品不满足需求（自然流失）、产品难以上手（受挫流失）和竞争产品影响（市场流失）。

  - 新用户要考虑如何在较少的数据支撑下做流失用户识别，提前防止用户流失，并如何对有效的新用户进行挽回。

  - 老用户流失：原因可能有到达用户生命周期衰退期（自然流失）、过度拉升arpu导致低端用户驱逐（刚性流失）、社交蒸发难以满足前期用户需求（受挫流失）和竞争产品影响（市场流失）。

  - 老用户有较多的数据，更容易进行流失用户识别，做好防止用户流失更重要。当用户流失后，要考虑用户生命周期剩余价值，是否需要进行挽回。

  - 参考@[王玮](https://www.zhihu.com/search?q=王玮&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"article"%2C"sourceId"%3A67650146}) 的回答：[如何进行用户流失原因调研？2100 关注 · 19 回答问题](https://www.zhihu.com/question/26225801)

**我们有一款游戏收入下降了，你怎么分析。**

- 两层模型：细分用户、渠道、产品，看到底是哪里的收入下降了

- 指标拆解：收入 = 玩家数量 * 活跃占比 * 付费转化率 * 付费次数 * 客单价

- 进一步细分，如玩家数量 = 老玩家数量 * 活跃度 + 新玩家数量 * 留存率等。然后对各个指标与以往的数据进行对比，发现哪些环节导致收入下降

- 原因分析：

  - a. 内部：产品变化、促活活动、拉新活动、定价策略、运营策略、服务器故障等

  - b. 外部：用户偏好变化、市场环境变化、舆论环境变化、竞争对手行为、外部渠道变化等

- 如何提高：基于乘法模型，可以采用上限分析，从前往后依次将指标提升到投入足够精力（假设优先分配人力、经费与渠道）后的上限，然后分析“收入”指标的数值提升。找到数值提升最快的那个阶段，就是我们提高收入的关键任务



**现在有一个游戏测试的环节，游戏测试结束后需要根据数据提交一份PPT，这个PPT你会如何安排？包括什么内容？**

- 这里可以套AARRR模型：

  - 获取用户（Acquisition）

  - 提高活跃度（Activation）

  - 提高留存率（Retention）

  - 获取收入（Revenue）

  - 自传播（Refer）

- 获取：我们的用户是谁？用户规模多大？
  - a. 用户分层

- 激活：游戏是否吸引玩家？哪个渠道获取的用户有质量（如次日留存高、首日停留时间长等）？

- 留存：用户能否持续留存？哪些用户可以留存？

- 转化：用户的游戏行为如何？能否进行转化？能否持续转化？

- 自传播：用户是否会向他人推荐该游戏？哪种方式能有效鼓励用户推荐该游戏？传播k因子是否大于1？



**某业务部门在上周结束了为期一周的大促，作为业务对口分析师，需要你对活动进行一次评估，你会从哪几方面进行分析?**

- （1） 确定大促的目的：拉新？促活？清库存？

- （2） 根据目的确定核心指标。

- （3） 效果评估：

  - a. 自身比较：活动前与活动中比较

  - b. 与预定目标比

  - c. 与同期其它活动比

  - d. 与往期同类活动比

- （4）持续监控：

  - a. 检查活动后情况，避免透支消费情况发生

  - b. 如果是拉新等活动，根据后续数据检验这批新客的质量





## 数据可视化

### 基础概念

**Exploratory Data Analysis (EDA)** is an approach of analyzing datasets to summarize their main characteristics, often using statistical graphics and other data visualization methods

> 缺失、imbalanced、feature、sample size、distribution、skews等等

#### Data types

- `Quantitative/numerical continuous` - 1, 3.5, 100, 10^10, 3.14
- `Quantitative/numerical discrete` - 1, 2, 3, 4
- `Qualitative/categorical unordered` - cat, dog, whale
- `Qualitative/categorical ordered` - good, better, best
- `Date or time` - 09/15/2021, Jan 8th 2020 15:00:00
- `Text` - The quick brown fox jumps over the lazy dog



#### Aesthetics

- a quantiﬁable set of features that are mapped to the data in a graphic. 
- Describe every aspect of a given graphical element.

<img src="../images/(null)-20220724150704920.(null)" alt="img" style="zoom: 33%;" />



#### Scales

**mapping** between data values and aesthetic values：比如什么样的数字代表圆形、什么样的数值代表x轴的位置

<img src="../images/(null)-20220724150701851.(null)" alt="img" style="width: 50%;" /><img src="../images/image.png" alt="image" style="width: 50%;" />

- Position scale

  - Cartesian coordinate system 笛卡尔坐标

  - Log transform需要注意x轴标签的问题
    - 场景：如果linear的话 无法capture所有的信息

<img src="../images/(null)-20220724150719594.(null)" alt="img" style="width:50%;" /><img src="../images/486be031-b2f9-4038-a400-0269d5a0889b.png" alt="486be031-b2f9-4038-a400-0269d5a0889b" style="width:50%;" />

- Color scale

  - distinguish groups of data

  - Represent data values （sequential color scale）

  - Tool to highlight


#### Visualization Collections

- Amount

  - Barplot：Unordered category的时候要rank！
    - Grouped & Stacked barplot

  - Dotplot


- Distributions

  - Histogram

  - Kernel Density： 多类别的时候kernel density plots work better than histograms

  - Boxplot

  - Violinplot

  - Ridgelineplot

- Proportions

  - Pie charts

  - Stacked bars


- Side-by-side bars

#### XY relationships

- Scatterplots
- Bubble plots
- Scatterplot matrix
- Correlation coefficient
- Correlogram

#### Uncertainty

- Probability distribution

<img src="../images/(null)-20220724152051641.(null)" alt="img" style="zoom:25%;" />

### R语言的数据可视化

https://ab5q6faprt.feishu.cn/wiki/wikcndQAXpgjyCzgZxGwc0JDp2f

主题设置大全：

![ggplot2-theme-elements-reference-v2_hu8994090e1960a0a71878a3756da20076_580819_2000x2000_fit_lanczos_2](../images/ggplot2-theme-elements-reference-v2_hu8994090e1960a0a71878a3756da20076_580819_2000x2000_fit_lanczos_2.png)
