# 数据分析

常见Tradeoff：最大化哪一个：
- 收益ROI = LT × ARPU / CPA
- 规模DAU = DNU × LT + RDAU



## 各赛道指标体系

这个模块是适合用思维导图整理的part！用飞书来整理啦：[指标体系](https://x11awkb08o.feishu.cn/docs/doccn1BQjkvMv2QOezX6DLngx1e#)





## 分析方法

### 指标纵向下钻

可以通过人群异质性效果探查（HTE），找到策略未触达的细分人群

<center><img src="../images/DA_metrics_1.png" width="35%"/></center>

### 指标横向分层

- 基于核心KPI指标：比如低转化率人群、中等、高转化率人群，知道运营动作要对的人
- 基于人群特征：年龄职业等，其实这些对购买习惯也有影响
- 基于场景：比如不同渠道、界面

RFM——对于这些分层方法的一种交叉





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
