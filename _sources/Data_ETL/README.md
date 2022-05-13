# DE基础

## SQL执行问题

假设Outer表M个pages、Inner表N个pages、每个pages有T个tuples

### SQL join

#### **Nested Join**

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


#### **Indexed Nested Loops Join**

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


#### **Hash Join**

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