# 损失函数

## 3. Loss（损失函数）实验

### 3.1 EMD、Chamfer 和固定索引 L2

- PyTorch Light 相同结构 20 轮：EMD 82.68，纯 Chamfer 82.33，Chamfer + 0.1 固定索引 L2 为 82.22，最大差 0.46。
- Jittor 组合对照中，Chamfer + 0.1 L2 的 iMonotone/Affine Final 为 82.04/81.82，纯 Chamfer 为 81.26/80.66。
- 固定索引项保留输出点与目标点的索引关系；Chamfer 只表达集合最近邻关系，不能自动保证一一对应。

**经验**：损失名称不能代替函数体。需要确认是双向平方 Chamfer、固定索引监督还是 EMD，以及损失是否真正接入模型。局部损失下降不保证整云 CD、P2S 和 Final 同时改善。

### 3.2 匹配、表面和覆盖损失

项目还测试了 Hungarian（匈牙利匹配）、Sinkhorn（软最优传输）、表面辅助、排斥、L2 rescue、密度与覆盖相关目标。已有记录中，部分代理损失下降，但出现重复点、覆盖损失、方向失控或完整评分下降。

**经验**：匹配损失改善的是对应关系，表面损失改善的是离面误差，覆盖约束改善的是点集分布；三者不能用一个训练损失或单一局部指标代替。实验必须保留完整点云评测和两个 CD 方向。

**证据**：[A 榜实验记录](../experiments/a_board_atlas.md)、[研究范围与经验](../experiments/research_findings.md)、[Oracle 分析](../experiments/oracle_analysis.md)。



[返回经验目录](README.md) · [查看分类实验](../experiments/README.md)
