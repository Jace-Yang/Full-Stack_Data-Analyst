# 基础概念

## 数据分层

- 数据运营层 (ODS)
    - ODS: Operation Data Store 数据准备区, 也称为贴源层。数据仓库源头系统的数据表通常会原封不动的存储一份, 这称为ODS层, 是后续数据仓库加工数据的来源。

- 数据仓库层 (DW) DW数据分层, 由下到上为:
    - DWD: data warehouse details 细节数据层, 是业务层与数据仓库的隔离层。主要对ODS数据层做一些数据清洗和规范化的操作。
        - 数据清洗: 去除空值、脏数据、超过极限范围的
    - DWB: data warehouse base 数据基础层, 存储的是客观数据, 一般用作中间层, 可以认为是大量指标的数据层。
    - DWS: data warehouse service 数据服务层, 基于DWB上的基础数据, 整合汇总成分析某一个主题域的服务数据层, 一般是宽表。用于提供后续的业务查询, OLAP分析, 数据分发等。
        -用户行为, 轻度聚合
        - 主要对ODS/DWD层数据做一些轻度的汇总。
- 数据服务层/应用层 (ADS)
    - ADS: applicationData Service应用数据服务, 该层主要是提供数据产品和数据分析使用的数据, 一般会存储在ES、mysq|等系统中供线上系统使用。
    - 我们通过说的报表数据, 或者说那种大宽表, 一般就放在这里

数据库设计三范式
为了建立冗余较小、结构合理的数据库，设计数据库时必须遵循一定的规则。在关系型数据库中这种规则就称为范式。范式时符合某一种设计要求的总结。

- 第一范式：确保每列保持原子性，即要求数据库表中的所有字段值都是不可分解的原子值。
- 第二范式：确保表中的每列都和主键相关。也就是说在一个数据库表中，一个表中只能保存一种数据，不可以把多种数据保存在同一张数据库表中。
作用：减少了数据库的冗余
- 第三范式：确保每列都和主键列直接相关，而不是间接相关。







## 数据仓库

#### 存储：Hadoop/HDFS

HDFS 架构原理HDFS采用Master/Slave架构。

一个HDFS集群包含一个单独的NameNode和多个DataNode。

**NameNode**作为Master服务，它负责管理文件系统的命名空间和客户端对文件的访问。NameNode会保存文件系统的具体信息，包括文件信息、文件被分割成具体block块的信息、以及每一个block块归属的DataNode的信息。对于整个集群来说，HDFS通过NameNode对用户提供了一个单一的命名空间。

**DataNode**作为Slave服务，在集群中可以存在多个。通常每一个DataNode都对应于一个物理节点。DataNode负责管理节点上它们拥有的存储，它将存储划分为多个block块，管理block块信息，同时周期性的将其所有的block块信息发送给NameNode。

#### 计算：MapReduce

**Mapper**: 当你向MapReduce框架提交一个计算作业时，它会首先把计算作业拆分成若干个Map任务，然后分配到不同的节点上去执行，每一个Map任务处理输入数据中的一部分。

**Reducer**: 当Map任务完成后，它会生成一些中间文件，这些中间文件将会作为Reduce任务的输入数据。Reduce任务的主要目标就是把前面若干个Map的输出汇总并输出。



## 底层数据库

### 三段锁协议

**共享锁（share lock）**：共享 (S) 用于只读操作，如 SELECT 语句。

如果事务T对数据A加上共享锁后，则其他事务只能对A再加共享锁，不能加排他锁。获准共享锁的事务只能读数据，不能修改数据。

**排他锁（exclusive lock）**：用于数据修改操作，例如 INSERT、UPDATE 或 DELETE。确保不会同时同一资源进行多重更新。

如果事务T对数据A加上排他锁后，则其他事务不能再对A加任任何类型的封锁。获准排他锁的事务既能读数据，又能修改数据。

**作用：**利用三段锁，可以避免以下问题

- 丢失修改：两个事务T1和T2读入同一数据并修改，T2提交的结果破坏了T1提交的结果，导致T1的修改被丢失

- 读脏数据：事务T1对数据D进行修改，事务T2读取到了事务T1修改的数据，接着事务T1发生异常进行回滚，事务T2读取到的就叫做“脏数据"

- 不可重复读：不可重复读是指事务T1读取数据后，事务T2执行更新操作，使T1无法再现前一次读取结果

#### 



## 大数据



### 数据倾斜

**定义**：Hadoop能够进行对海量数据进行批处理的核心，在于它的分布式思想，也就是多台服务器（节点）组成集群，进行分布式的数据处理。但当大量的数据集中到了一台或者几台机器上计算，这些数据的计算速度远远低于平均计算速度，导致整个计算过程过慢，**这种情况就是发生了数据倾斜**。

**产生机制：**无论是MR还是Spark任务进行计算的时候，都会触发**Shuffle**动作。一些典型的操作如distinct、reduceByKey、groupByKey、join、repartition等都会触发shuffle：一旦触发，Spark就会将相同的key及其value拉到一个节点上。如果有某个key及其对应的数据太多的话，那就会发生明显的单点问题——单个节点处理数据量爆增的情况

- `Join`
  - 一个表很小但key集中：分发到某一个或几个Reduce/Stage上的数据远高于平均值
  - 大表与大表，但是分桶的判断字段0值或空值过多：空值由一个Reduce处理很慢
- `Group by`
  - 维度太小了 有些值处理太多，所以reduce很耗时
- `Count Distinct` / sum(distinct key)
  - 某特殊值过多，处理特殊值好时

**发生原因**：

- 数据频率倾斜——某一个区域的数据量要远远大于其他区域

  - **key分布不均匀**

    1、某些key的数量过于集中，存在大量相同值的数据

    2、存在大量异常值或空值。

  - 唯一值非常少，极少数值有非常多的记录值(唯一值少于几千)

  - 唯一值比较多，这个字段的某些值有远远多于其他值的记录数，但是它的占比也小于百分之一或千分之一

- 数据大小倾斜——部分记录的大小远远大于平均值。

**解决/缓和的办法**

- **通用：**提高shuffle并行度 `set spark.sql.shuffle.partitions= [num_tasks] ` （默认200），将原本被分配到同一个Task的不同Key分配到不同Task。

- `Group By`的时候：可以map端聚合，启动负载均衡：使计算变成了两个mapreduce

  - 在第一个中在 shuffle 过程 partition 时随机给 key 打标记，使每个key 随机均匀分布到各个 reduce 上计算，完成部分计算（但相同key没有分配到相同reduce上，所以需要第二次的mapreduce）
  - 第二次回归正常 shuffle，但数据分布不均匀的问题在第一次mapreduce已经有了很大的改善
  
    ```sql
    set hive.map.aggr=true; --在map中会做部分聚集操作，效率更高但需要更多的内存
    set hive.groupby.skewindata=true; --默认false，数据倾斜时负载均衡
    ```

  - 例子：

    ```sql
    --水果字段名为category
    select count (substr(x.category,1,2)) 
    from
    (select concat(category,'_',cast(round(10*rand())+1 as string))
    from table1
    group by concat(category,'_',cast(round(10*rand())+1 as string))
    ) x --1阶段聚合
    group by substr(x.category,1,2);   --2阶段聚合
    ```

- 小表Join大表的时候可以把Reduce Join改成Map Join

  - 原理：mapjoin优化就是在Map阶段完成join工作，而不是像通常的common join在Reduce阶段按照join的列值进行分发数据到每个Reduce上进行join工作。这样避免了**Shuffle**阶段，从而避免了数据倾斜。

  - 注意：这个操作会将所有的小表全量复制到每个map任务节点，然后再将小表缓存在每个map节点的内存里与大表进行join工作。小表的大小的不能太大，一般也就几百兆，否则会出现OOM报错。

  - 代码

    ```sql
    set hive.auto.convert.join = true; -- hive是否自动根据文件量大小，选择将common join转成map join 。
    set hive.mapjoin.smalltable.filesize =25000000; --大表小表判断的阈值，如果表的大小小于该值25Mb，则会被判定为小表。则会被加载到内存中运行，将commonjoin转化成mapjoin。一般这个值也就最多几百兆的样子。
    ```

- **开启Skewed Join**

  hadoop 中默认是使用hive.exec.reducers.bytes.per.reducer = 1000000000

  也就是每个节点的reduce 默认是处理1G大小的数据，如果join操作也产生了数据倾斜，可以在hive 中设定

  ```sql
  set hive.optimize.skewjoin = true;
  set hive.skewjoin.key = skew_key_threshold （default = 100000）
  ```

  - 建议每次运行比较复杂的sql 之前都可以设一下这个参数. 如果你不知道设置多少，可以就按官方默认的1个reduce 只处理1G 的算法，那么 skew_key_threshold = 1G/平均行长
  - 或者默认直接设成250000000 (差不多算平均行长4个字节)

  当我们开启Skew Join之后，在运行时，会对数据进行扫描并检测哪个key会出现倾斜，对于会倾斜的key，用map join做处理，不倾斜的key正常处理。

- 数据打散（**重新设计key**）

  - 在map阶段时给key加上一个随机数，有了随机数的key就不会被大量的分配到同一节点(小几率)，待到reduce后再把随机数去掉即可；
  - 将大表(A)中的id加上后缀(即“id_0”-“id_2”)，起到“打散”的作用。为了结果正确，小表B中的id需要将每条数据都“复制”多份。此时再执行join操作，将会产生三个task，每个task只需要关联一条数据即可，起到了分散的作用<img src="../images/86559.png" alt="86559" style="zoom:50%;" />
    - `SELECT id, value, concat(id, round(rand() * 10000)%3) as new_id`
  - 经过处理之后再使用new_id来作为聚合条件

- 大表Join大表的时候可以大表拆分，倾斜部分单独处理（比如非常多的卖家、但很多卖家的订单很少）

  - 中间表分桶排序后join

- partition

  - 比如现在是按省份进行汇总数据，如果只是简单的按省份去分（这并没有错），那么数据肯定会倾斜，因为各省的数据天然不一样。我们可以通过历史数据、抽样数据或者一些常识，对数据进行人工分区，让数据按照我们自定义的分区规则比较均匀的分配到不同的task中。

  - 常见的分区方式：

    - 随机分区：每个区域的数据基本均衡，简单易用，偶尔出现倾斜，但是特征同样也会随机打散。

    - 轮询分区：绝对不会倾斜，但是需要提前预知分成若干份，进行轮询。

    - hash散列：可以针对某个特征进行hash散列，保证相同特征的数据在一个区，但是极容易出现数据倾斜。

    - 范围分区：需要排序，临近的数据会被分在同一个区，可以控制分区数据均匀。

- 开性能

  - **增加jvm**（Java Virtual Machine：Java虚拟机）**内存**，这适用于变量值非常少的情况，这种情况下，往往只能通过硬件的手段来进行调优，增加jvm内存可以显著的提高运行效率；
  - **增加reduce的个数**，这适用于变量值非常多的情况，这种情况下最容易造成的结果就是大量相同key被partition到一个分区，从而一个reduce执行了大量的工作；


## 参考资料：

- [数据仓库分层中的ODS、DWD、DWS ](https://www.cnblogs.com/amyzhu/p/13513425.html)
- [知乎｜逆流君｜漫谈数据倾斜解决方案（干货）](https://zhuanlan.zhihu.com/p/332368318)