# 统计问题

### 辛普森悖论 Simpson's Paradox

统计层面的核心是：Partial correlation and marginal correlation can be dramatically different

- [例1]: 男性用药多 但恢复率更高，所以让整体有这个correlation，不过by group的correlation不是这样！
    <center><img src="../../images/CI_simpson_paradox.png" width="60%"/></center>

- [例2]: 比如说Y是购买概率，X是看直播时长，有可能实际上 男生女生都是看得越久 购买的概率越低，但由于普遍女生看的久，购买概率高，在整体数据上会体现出“观看直播时间越长，购买概率越高”的错误结论！
    