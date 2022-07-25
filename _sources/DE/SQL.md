# SQL

## SQL优化

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