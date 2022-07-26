# SQL



## 数据类型

NULL

- NULL 与任何值的直接比较都为 NULL。
- NULL<>NULL 的返回结果是 NULL， 而不是 false。
- NULL=NULL 的返回结果是 NULL， 而不是 true。
- NULL<>1 的返回结果是 NULL，而不是 true。

## 索引

主键索引：必须要有主键，主键建议是单列字段，使用有序递增的整数值做主键。

- 如果没有主键，主库执行默认加一个隐式主键，所以主库执行会比较快。但从库并不是解析原sql，而是回放row模式的binlog，此时就是全表扫描，会非常慢，从而导致主从延迟。
- 不要以为唯一索引影响了 insert 速度，这个速度损耗可以忽略，但提高查找速度是明显的；另外，即使在应用层做了非常完善的校验控制，只要没有唯一索引，必然有脏数据产生。

## SQL优化

#### Join的问题

假设Outer表M个pages、Inner表N个pages、每个pages有T个tuples

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

#### 其他

注释

- 必要时在 SQL 代码中加入注释。优先使用 C 语言式的以 /* 开始以 */ 结束的块注释。
  - 说明：标准 SQL 中，注释以两个短划线（--）开头，以新行结束，但这是因为第一个 SQL 引擎是在 IBM 大型机上的，使用的是穿孔卡。在可以存储自由格式文本的现代计算机



## 写SQL





#### [180. 连续出现的数字](https://leetcode-cn.com/problems/consecutive-numbers/)

```SQL
WITH Num_Series_id AS(
    SELECT Num,
            (row_number() OVER (ORDER BY id ASC) - 
                row_number() OVER (PARTITION BY Num ORDER BY id ASC)) AS series_id -- 连续出现的数特点为：[行号] - [组内行号] = k*/
    FROM Logs)

SELECT DISTINCT Num "ConsecutiveNums"
FROM Num_Series_id
GROUP BY Num, series_id
HAVING COUNT(1) >= 3  -- 连续重复次数
```

#### [511. 游戏玩法分析 I](https://leetcode-cn.com/problems/game-play-analysis-i/)

```SQL
SELECT player_id, min(event_date) AS first_login
FROM Activity
-- WHERE games_played = 1
GROUP BY player_id
```



#### [512. 游戏玩法分析 II](https://leetcode-cn.com/problems/game-play-analysis-ii/)

- 开窗筛选法

```SQL
WITH Activity_with_num AS(

    SELECT

        player_id,

        device_id,

        ROW_NUMBER() OVER(PARTITION BY player_id ORDER BY event_date) AS login_n

    FROM

        Activity

)
SELECT player_id, device_id

FROM Activity_with_num

where login_n = 1
```

- Left  join法

```SQL
WITH Activity_with_num AS(

    SELECT

        player_id,

        min(event_date) AS first_date

    FROM

        Activity

    GROUP BY

        player_id

)
SELECT a.player_id, b.device_id

FROM Activity_with_num a 

    LEFT JOIN Activity b 

        ON a.player_id = b.player_id

        AND a.first_date = b.event_date
```

- where法

```SQL
WITH Activity_with_num AS(

    SELECT

        player_id,

        min(event_date) AS first_date

    FROM

        Activity

    GROUP BY

        player_id

)
SELECT player_id, device_id

FROM Activity 

WHERE (player_id, event_date) IN (SELECT * FROM Activity_with_num)
```

#### [534. 游戏玩法分析 III](https://leetcode-cn.com/problems/game-play-analysis-iii/)

```SQL
SELECT 

    player_id,

    event_date,

    SUM(games_played) over(partition by player_id order by event_date) as games_played_so_far

FROM

    Activity
```



#### [550. 游戏玩法分析 IV](https://leetcode-cn.com/problems/game-play-analysis-iv/)

```SQL
With temp as(
    select
    player_id,
    event_date,
    -- 这里获取了每一个用户的 第一次登录时间与当前这条的 时间差值
    dateDiff(event_date, min(event_date) over(partition by player_id order by event_date asc) ) as diff
    from Activity
    )
select
ROUND (
-- 这里使用case when ，只把与第一天登录相差一天的登录用户加入计算
sum( case when diff = 1 then 1 else 0 end ) / count(distinct player_id)
,2
) AS fraction
from temp
```



#### [1045. 买下所有产品的客户](https://leetcode-cn.com/problems/customers-who-bought-all-products/)

```SQL
# Write your MySQL query statement below

SELECT customer_id
FROM Customer
GROUP BY customer_id
HAVING count(distinct product_key) = (SELECT count(product_key) FROM Product)
```



#### [1097. 游戏玩法分析 V](https://leetcode-cn.com/problems/game-play-analysis-v/)

```SQL
-- 每个用户先统计一张登陆流水，于此同时打上注册日的tag
WITH user as (
    SELECT 
        player_id, 
        event_date,
        min(event_date) over(partition by player_id) AS first_login 
    FROM 
        Activity 
)
-- 把第二天有登录的给算出来
SELECT 
    first_login as install_dt,
    count(distinct player_id) as installs,
    round(
        sum(if(datediff(event_date, first_login) = 1, 1, 0)) / count(distinct player_id),
        2
        ) AS Day1_retention
FROM user
GROUP BY first_login -- 算次留存的细节：可以groupby首登/注册日期
```

#### [1843. 可疑银行账户](https://leetcode-cn.com/problems/suspicious-bank-accounts/)

```SQL
WITH oversave_log AS (
    SELECT 
        account_id, 
        DATE_FORMAT(day, '%Y%m') AS yearmonth
    FROM 
        Transactions JOIN Accounts USING (account_id)
    WHERE 
        type = 'Creditor'
    GROUP BY 
        account_id, DATE_FORMAT(day, '%Y%m')
    HAVING 
        SUM(amount) > AVG(max_income)
)

SELECT DISTINCT account_id
FROM oversave_log
WHERE (account_id, PERIOD_ADD(yearmonth, 1)) IN (SELECT * FROM oversave_log)
```



#### 查询连续X天登陆的用户

```SQL
-- 每个用户先统计一张登陆流水，日期转换int
WITH user as (
    SELECT 
        player_id, 
        DATEDIFF(event_date, '2000-01-01') AS date_id
    FROM 
        Activity 
    GROUP BY
        player_id, 
        date_id
    ORDER BY date_id
),

-- 然后把累加开始的时间当作每个用户的「区间开始」id
user_cum_tagged AS(
    SELECT 
        player_id,
        date_id - row_number() over(PARTITION by player_id order by date_id) as cum_id
    FROM user
)

-- 每个用户、每个「区间开始」id组中 有2条登录天数的即为连续2天登录，然后把「区间开始」id转化为实际日期
SELECT player_id, DATE_ADD('2000-01-01', INTERVAL cum_id+1 DAY) AS start_date
    -- 这里注意-row_number的时候第一天也会减1，所以这里加回来
FROM user_cum_tagged
GROUP BY player_id, DATE_ADD('2000-01-01', INTERVAL cum_id+1 DAY)
HAVING COUNT(*) >= 2 -- 这里可以改成任意天～
```



#### 上周流失用户

每个用户登录流水中**max(date) <=上周最后一个日期** 