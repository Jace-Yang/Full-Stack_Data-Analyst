# 基础

## 机器学习分类

<img src="../images/(null)-20220724132314070.(null)" alt="img" style="zoom: 33%;" />

- 根据label的类型来分

  - `Supervised learning ` algorithms： Learn a function that maps inputs to an output from a set of **labeled** training data.
    - Downside: getting labeled data is pretty costly. Hard to have clean labeled data.
  - `Unsupervised Learning`: learn patterns from **unlabeled** data samples.
  - `Reinforcement Learning`: The computer employs trial and error to come up with a solution to the problem. To get the machine to do what the programmer wants, the artificial intelligence **gets either rewards or penalties** for the **actions** it performs. Its goal is to **maximize the total reward**.
  - `Deep Learning`: a class of ML algorithms that uses multiple layers to progressively extract **higher-level features/abstractions** from raw inputs.
    - 本质上也是create 很多features that you think is important

- **其他特定的学习方式**

  - `Active Learning`：不是exposed to all training data, but we can ask for the label

    - 可以减少对label的依赖！

    - Create a human annotator, and makes tha ML ask for human to label

    >  https://en.wikipedia.org/wiki/Active_learning_(machine_learning)
    >
    > A learning algorithm can interactively query a user (or some other information source) to **label new data points with the desired outputs**

  - `Self-supervised Learning`: Trying to predict a label based on unlabeled data

    - Self-supervised learning techniques are generally used as pre-train tasks to *generate labels for other downstream tasks*.
      - e.g 补全单词预测——因为补全句子这种你用来补全的context words也不算labeled
      - word2vec，autoencoder这类明明没标签却能够造出目标函数，使用凸优化方法求解，无中生有
    - The learning model **trains itself by leveraging one part of the data** to **predict the other part** and generate labels accurately. In the end, this learning method ***converts an unsupervised learning problem into a supervised one***

    - **区分定义**

      - Self-supervised learning vs semi-supervised learning：<u>自监督是完全没标签，半监督其实就是有部分标签的时候带了一些数据生产规则的有监督</u>

        - Semi-supervised learning uses **manually** labeled training data for supervised learning and unsupervised learning approaches for unlabeled data to generate a model that leverages existing labels but builds a model that can **make predictions beyond the labeled data.** 

          - > **一个semi-supervised的**[例子](https://zhuanlan.zhihu.com/p/33196506)：简单自训练**（simple self-training）：用有标签数据训练一个分类器，然后用这个分类器对无标签数据进行分类，这样就会产生伪标签（pseudo label）或软标签（soft label），挑选你认为分类正确的无标签样本（此处应该有一个**挑选准则**），把选出来的无标签样本用来训练分类器。

        - Self-supervised learning **relies completely on data** that lacks manually generated labels.

      - Self-supervised learning vs unsupervised learning：<u>自监督的目标是监督学习，无监督不是</u>

        - Self-supervised learning is similar to unsupervised learning because both techniques work with datasets that don’t have manually added labels. In some sources, self-supervised learning is addressed as a subset of unsupervised learning. However, unsupervised learning **concentrates on clustering, grouping, and dimensionality reduction**, 
        - while self-supervised learning aims to draw conclusions for regression and classification tasks.

      - Hybrid Approaches vs. Self-supervised Learning：<u>自监督学习和人类混合的差别是自监督是全自动的</u>

        - There are also hybrid approaches that combine automated data labeling tools with supervised learning. In such methods, computers can label data points that are easier-to-label relying on their training data and **leave the complex ones to humans.** Or, they can label all data points automatically but need human approval. 
        - In self-supervised learning, **automated data labeling is embedded** in the training model. The *dataset is labeled as part of the learning processes*; thus, it doesn’t ask for human approval or only label the simple data points. 

  - `Transfer Learning`：pre-trained models are used as the starting point on computer vision and natural language processing tasks

## 模型的评测指标

metric主要用来评测机器学习模型的好坏程度，不同的任务应该选择不同的评价指标。分类，回归和排序问题应该选择不同的评价函数。

Evaluation Metrics

- Evaluation metrics are generally used to measure the performance of an ML model.
- Evaluation metrics indicate how well the models would do when deployed
- The choice of metrics is very task-speciﬁc and determines what the model learns
  - can direct your model to learn specific things based on the evaluation metric
- It is important to know what you are willing to **trade oﬀ** when training ML models for a task

### **回归指标**

- 平均绝对误差(MAE)，又称L1范数损失：

  $$MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|$$

  - MAE虽能较好衡量回归模型的好坏，但是绝对值的存在导致**函数不光滑**，在某些点上不能求导，
  - 可以考虑将绝对值改为残差的平方，这就是均方误差。

- 均方误差(MSE)，又被称为 L2范数损失 。

  $$M S E=\frac{1}{n} \sum_{i=1}^{n}\left(y_{i}-\hat{y}_{i}\right)^{2}$$

  - 由于MSE与我们的目标变量的量纲不一致，为了保证量纲一致性，我们需要对MSE进行开方 ：

- 均方根误差(RMSE)

  $$RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2} $$

- $R^{2}$又称决定系数，表示反应因变量的全部变异能通过数学模型被自变量解释的比例

  $$R^{2} =1-\frac{\sum^n_{i}\left(y_{i}-\hat{y}\right)^{2} / n}{\sum^n_{i}\left(y_{i}-\bar{y}\right)^{2} / n}=1-\frac{M S E}{\operatorname{Var}}  $$

  - $y$表示实际销量, $\hat{y}$表示预测销量, $\bar{y}$表示实际销量的均值, $n$表示样本数, $i$表示第$i$个样本, $Var$表示实际值的方差，也就是销量的变异情况。

  - $R2_{score}$越大，模型准确率越好。

- $MSE$表示均方误差，为残差平方和的均值,该部分不能能被数学模型解释的部分,属于不可解释性变异。
  - 因此： $$可解释性变异占比 = 1-\frac{不可解释性变异}{整体变异}= 1-\frac{M S E}{\operatorname{Var}} = R2_score \tag{5} $$

### 分类指标

**分类**

- Threshold-based metrics
  - Classiﬁcation Accuracy
  - Precision, Recall & F1-score
- Ranking-based metrics
  - Average Precision (AP)
  - Area Under Curve (AUC)

准确率和错误率

$$Acc(y,\hat{y})=\frac{1}{n}\sum_{i=1}^{n}y_i=\hat{y_i} $$

$$ Error(y, \hat{y})=1-acc(y,\hat{y}) \tag{7} $$ 

- Acc与Error平等对待每个类别，即**每一个样本判对 (0) 和判错 (1) 的代价都是一样的**。
- 使用Acc与Error作为衡量指标时，需要考虑样本不均衡问题以及实际业务中好样本与坏样本的重要程度。



**混淆矩阵**：对于二分类问题,可将样例根据其真是类别与学习器预测类别的组合划分为

```PHP
真正例(true positive, TP):预测为 1，预测正确，即实际 1

假正例(false positive, FP):预测为 1，预测错误，即实际 0

真反例(ture negative, TN):预测为 0，预测正确，即实际 0

假反例(false negative, FN):预测为 0，预测错误，即实际 1
```

则有:TP+FP+TN+FN=样例总数

- 准确率（查准率） Precision：分类器**预测的正样本中预测正确的比例**，取值范围为[0,1]，取值越大，模型预测能力越好。 $$ P=\frac{TP}{TP+FP} \tag{8} $$

- 召回率（查全率）Recall：Recall 是分类器所预测正确的正样本占所有正样本的比例，取值范围为[0,1]，取值越大，模型预测能力越好。 $$ R=\frac{TP}{TP+FN} \tag{9} $$

- F1 Score：Precision和Recall 是互相影响的，理想情况下肯定是做到两者都高，但是一般情况下Precision高、Recall 就低， Recall 高、Precision就低。为了均衡两个指标，我们可以采用Precision和Recall的加权调和平均（weighted harmonic mean）来衡量，即

   $$ \frac{1}{F_1}=\frac{1}{2} \cdot (\frac{1}{P}+\frac{1}{R}) \tag{10} $$, $$ F_1=\frac{2*P*R}{P+R} \tag{11} $$

- ROC，全称是"受试者工作特征"(Receiver Operating Characteristic)曲线，ROC曲线为 FPR 与 TPR 之间的关系曲线，这个组合以 FPR 对 TPR，即是以代价 (costs) 对收益 (benefits)，显然收益越高，代价越低，模型的性能就越好。

  - ROC曲线的横轴是"假正例率"(False Positive Rate, **FPR**), 纵轴是"真正例率"(True Positive Rate, **TPR**), **注意这里不是上文提高的P和R**.

  - y 轴为真阳性率（TPR）：在所有的正样本中，分类器预测正确的比例（等于Recall）

    $$ TPR=\frac{TP}{TP+FN} \tag{12} $$

  - x 轴为假阳性率（FPR）：在所有的负样本中，**分类器预测错误的比例**

    $$ FPR=\frac{FP}{TN+FP} \tag{13} $$

  现实使用中,一般使用有限个测试样例绘制ROC曲线,此时需要有有限个(真正例率,假正例率)坐标对. 绘图过程如下:

  - 给定$m^+$个正例和$m^-$个反例,根据学习器预测结果对样例进行排序,然后将分类阈值设为最大,此时真正例率和假正例率都为0,坐标在(0,0)处,标记一个点.
  - 将分类阈值依次设为每个样本的预测值,即依次将每个样本划分为正例.
  - 假设前一个坐标点是(x,y),若当前为真正例,则对应坐标为$(x,y+\frac{1}{m^+})$, 若是假正例,则对应坐标为$(x+\frac{1}{m^-}, y)$
  - 线段连接相邻的点.

- AUC：对于二分类问题，预测模型会对每一个样本预测一个得分s或者一个概率p。 然后，可以选取一个阈值t，让得分s>t的样本预测为正，而得分s<t的样本预测为负。 这样一来，根据预测的结果和实际的标签可以把样本分为4类,则有混淆矩阵：

  - 随着阈值t选取的不同，这四类样本的比例各不相同。定义真正例率TPR和假正例率FPR为： $$ \begin{array}{l} \mathrm{TPR}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}} \ \mathrm{FPR}=\frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}} \end{array} \tag{14} $$ 随着阈值t的变化，TPR和FPR在坐标图上形成一条曲线，这条曲线就是ROC曲线。 显然，如果模型是随机的，模型得分对正负样本没有区分性，那么得分大于t的样本中，正负样本比例和总体的正负样本比例应该基本一致。
  - 实际的模型的ROC曲线则是一条上凸的曲线，介于随机和理想的ROC曲线之间。而ROC曲线下的面积，即为AUC！
  - 这里的x和y分别对应TPR和FPR，也是ROC曲线的横纵坐标。 $$ \mathrm{AUC}=\int_{t=\infty}^{-\infty} y(t) d x(t) \tag{15} $$



#### 对比

首先，两个都是both are ranking metrics：如果一个模型预测的probability是之前的一半，AUC和AP都不会改变！log-loss之类的才会改变

<img src="../images/(null)-20220726015315694.(null)" alt="img" style="zoom:50%;" />



- **Precision-Recall PR Curve ⇒ Average Precision(AP)**

  - 好处是imbalance datasets表现更好make more sense： In case of imbalance datasets, AP is a better estimate indicative of model

    - AUC will still be very high even the model is bad! 没给你true picture

      <img src="../images/(null)-20220726015315595.(null)" alt="img" style="zoom: 50%;" />

- Receiver Operating Curve(ROC) ⇒ Area Unver ROC (AUROC / AUC)

  - AUROC measures whether the model is able to rank positive examples higher than negative samples 
  - 根据probability来rank prediction：如果全对的话，就是正确地把positive都放上面了，这样从threshold=1 recall=1 慢慢下降threshold的过程中，没有False Positive出现所以在FPR=0往上走，然后等threshold卡完所有positive继续往下走才因为threshold过低出现False Positive，但这个时候所有的P positive都被检测了，所以在recall = 1往右走
  - 好处是有Benchmark: random prediction ⇒ 0.5 
    - easier to know how well the model is performing than random using AUROC than AP 
    - 因为这样移动threshold的时候会按sample的比例一边丢一个
    - If you get a score of 0 that means the classifier is perfectly incorrect, it is predicting the incorrect choice 100% of the time.

- KS值(Kolmogorov-Smirnov)是在模型中用于**区分预测正负样本分隔程度**的评价指标，一般应用于金融风控领域。

  与ROC曲线相似，ROC是以FPR作为横坐标，TPR作为纵坐标，通过改变不同阈值，从而得到ROC曲线。

  ks曲线为TPR-FPR，ks曲线的最大值通常为ks值。可以理解TPR是收益，FPR是代价，ks值是收益最大。图中绿色线是TPR、蓝色线是FPR。

  KS的计算步骤如下：

  1. 按照模型的结果对每个样本进行打分
  2. 所有样本按照评分排序，从小到大分为10组（或20组）
  3. 计算每个评分区间的好坏样本数。
  4. 计算每个评分区间的累计好样本数占总好账户数比率(good%)和累计坏样本数占总坏样本数比率(bad%)。
  5. 计算每个评分区间累计坏样本占比与累计好样本占比差的绝对值（累计bad%-累计good%），然后对这些绝对值取最大值即得此评分模型的K-S值。

- CTR（Click-Through-Rate）

  CTR即点击通过率,是互联网广告常用的术语,指网络广告（图片广告/文字广告/关键词广告/排名广告/视频广告等）的点击到达率,即该广告的实际点击次数（严格的来说,可以是到达目标页面的数量）除以广告的展现量(Show content). $$ ctr=\frac{点击次数}{展示量}　\tag{16} $$

- CVR (Conversion Rate)，CVR即转化率。是一个衡量CPA广告效果的指标

  简言之就是用户点击广告到成为一个有效激活或者注册甚至付费用户的转化率. $$ cvr=\frac{点击量}{转化量}　\tag{17} $$



#### 多分类

多分类时都一样，除了AUC

<img src="../images/(null)-20220726015221534.(null)" alt="img" style="zoom:50%;" />



## 参考

1. 周志华 西瓜书
2. 李航 统计学习方法
3. https://baike.baidu.com/item/CVR/20215345
4. https://baike.baidu.com/item/CTR/10653699?fr=aladdin
5. https://www.cnblogs.com/shenxiaolin/p/9309749.html