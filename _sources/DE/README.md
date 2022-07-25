# 基础概念

##### 数据分层

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


## SQL各种问题

假设Outer表M个pages、Inner表N个pages、每个pages有T个tuples


### 数据类型

NULL
- NULL 与任何值的直接比较都为 NULL。
- NULL<>NULL 的返回结果是 NULL， 而不是 false。
- NULL=NULL 的返回结果是 NULL， 而不是 true。
- NULL<>1 的返回结果是 NULL，而不是 true。

### 索引

主键索引：必须要有主键，主键建议是单列字段，使用有序递增的整数值做主键。
- 如果没有主键，主库执行默认加一个隐式主键，所以主库执行会比较快。但从库并不是解析原sql，而是回放row模式的binlog，此时就是全表扫描，会非常慢，从而导致主从延迟。
- 不要以为唯一索引影响了 insert 速度，这个速度损耗可以忽略，但提高查找速度是明显的；另外，即使在应用层做了非常完善的校验控制，只要没有唯一索引，必然有脏数据产生。

### SQL优化

#### Join的问题

我们需要限制 JOIN 个数
- 单个 SQL 语句的表 JOIN 个数不能超过 5 个
- 过多的 JOIN 个数会带来极低的查询效率
- 过多 JOIN 在编译执行计划时消耗很大

Join的类型：
- **Nested Join**

    场景：没有index左右表被join的column的时候使用

    ```
    # Given Outer的一行：挨个去inner的page的row里面找相等的就吐出来
    def joinrow(orow, ipage): 
        for irow in ipage: 
            if orow.p == irow.p: 
                yield (orow, irow)
    
    # Given Outer的page：里面Outer的每一行 挨个去跑上面的算法找到ipage里面的matched的
    def joinpages(opage, ipage): 
        for orow in opage: for resulttuple in joinrow(orow, ipage): 
            yield resulttuple 
            
    
    for opage in outer:      # need to read from disk
        for ipage in inner:  # need to read from disk
            joinpages(opage,ipage)
    ```

    - Cost: M (for each outer page) + MN (fetch all inner pages for each outer page) 

    因此要尽量小表Join大表：Always choose the **smaller** of the two relations as outer relation
    > A JOIN B, A 是outer表 B是inner表

    - 写在关联左侧的表每有1条重复的关联键时底层就会多1次运算处理
    - 把重复关联键少的表放在join前面做关联可以提高join的效率


- **Indexed Nested Loops Join**

    假设
    - outer的predicate有5%的selectivity
    - inner的join attribute p已经被index了,lookup index的cost 是$C_1$
    ```
    for opage in outer:                     # read from disk
        for orow in opage:                  # in memory
            for ipage in index.get(orow.p): # read from disk
                joinrow(orow, ipage)        
    ```

    总共的cost = M + T × M × 0.05 × $C_1$
    > M (for each outer page) + T * M * 0.05 * C1 (look up index for each outer tuple)
    - 第四行局部cost=1个page
    - 第三行局部cost= (1个orow返回出C的概率0.05 × lookup C的ipage的cost $C_1$ ) × 1
    - 第二行局部cost= 1 + (0.05 $C_1$) × 一个opage的tuple总数T
    - 第一行总cost= M × (1 + 0.05 × T × $C_1$) 


- **Hash Join**

    假设是
    - Equality joins：必须是有foreign key constrain的那种join
    - Hash table in memory, assume no overflow pagesà1 lookup to get tuple

    ```
    index = initialize hash index 
    for ipage in inner: 
        for irow in ipage: 
            index.insert(irow.p, irow) 
    for opage in outer: 
        for orow in opage: 
            for irow in index.get(orow.p): 
                yield (row, irow)
    ```

    总cost：N + M + T × M
    > N (build hash index for inner pages) + M (for each outer page) + (T * M) * 1 (look up index for each outer tuple)
    - 建index的cost：N个pages
    - outer里面搜joined的cost: M + M × T
        - 第四行：1
        - 第三行：平均1个出来的irow × 1
        - 第二行：读取1个Opage + 一个opage里面T个tuple × 1
        - 第一行：M个 page × (1 + T)


NPE(Null Pointer Exception)问题：

- 当某一列的值全是 NULL 时， count(col)的返回结果为 0，但 sum(col)的返回结果为 NULL，因此使用 sum()时需注意 NPE 问题。

    ```sql
    SELECT IF(ISNULL(SUM(g)),0,SUM(g))
    FROM table;
    ```

    可以避免
### 其他

注释
- 必要时在 SQL 代码中加入注释。优先使用 C 语言式的以 /* 开始以 */ 结束的块注释。
    - 说明：标准 SQL 中，注释以两个短划线（--）开头，以新行结束，但这是因为第一个 SQL 引擎是在 IBM 大型机上的，使用的是穿孔卡。在可以存储自由格式文本的现代计算机


## 参考资料：

- [数据仓库分层中的ODS、DWD、DWS ](https://www.cnblogs.com/amyzhu/p/13513425.html)