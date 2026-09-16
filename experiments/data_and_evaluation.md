# 数据、采样与评测

[分类目录](README.md) · [对应经验](../lessons/data_and_evaluation.md)

A 榜逐条记录见 [完整实验记录](a_board_atlas.md) 中的 `A-FND`、`A-DATA` 组。该分类收录 10 条主记录；同一记录的交叉引用不另计实验。

## 1. 任务与评测

输入为带噪三维点云，输出点数与输入一致。评分包含 CD（倒角距离）和 P2S（点到表面的距离）两项，Final 为两项得分的平均值。最终提交使用 Jittor 实现，推理不读取测试集干净点云、网格或标签。

方法比较使用完整点云评测结果，同时记录训练方式、数据范围和计算成本。局部块损失与位移统计作为诊断指标单独报告。



## 数据对照与复现入口

A 榜全量覆盖与分片轮转记录在 `A-DATA` 组；B 榜训练对生成、网格筛查、稳定随机种子与 MOCK 的坐标变换见 [评测规范](../docs/evaluation_protocol.md) 和 [B 榜复现指南](../b_board/reproduction.md)。

MOCK 是本地研发评测，不能默认解释为独立新形状泛化测试。对形状隔离的结论，应使用明确排除重叠的训练/评测清单。

## 证据与范围

[方法比较](evidence/method_comparison_evidence.json)、[迁移摘录](evidence/jittor_adaptation_evidence.json)、[研究摘录](evidence/research_evidence.json) 和 [Oracle 摘录](evidence/oracle_evidence.json) 保留指标及来源哈希。分类页面是已有记录的整理，不增加独立实验数。完整历史日志、缓存、诊断程序和检查点未全部公开。
