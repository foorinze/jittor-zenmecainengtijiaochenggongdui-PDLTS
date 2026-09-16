# 实验记录

按研究问题查询设置、对照、结果和采用情况。先看 [方案选择总览](overview.md)，再进入各类记录；经验归纳单独放在 [经验总结](../lessons/README.md)。

| 分类 | 主要内容 |
|---|---|
| [方法横评与 Light/Heavy](method_comparison.md) | A-SELECT |
| [数据、采样与评测](data_and_evaluation.md) | A-FND、A-DATA |
| [损失与监督目标](losses.md) | A-LOSS、A-ANCHOR、A-ASSIGN |
| [结构与表示](architecture.md) | A-ARCH |
| [训练、级联与 A/B 最终配置](training_and_cascade.md) | A-CASCADE |
| [几何修正与位移学习](geometry.md) | A-BRIDGE |
| [Oracle 与集合选择](oracle_and_selection.md) | A-TRANSPORT |
| [迁移、配置接线与数值检查](engineering.md) | 跨类别的实现与数值诊断 |

## A/B 榜与后期研究

- [A 榜完整记录](a_board_atlas.md)：86 条稳定编号的历史条目，包含评测、诊断、负结果与交叉索引，不将条目数视作独立实验数。
- [B 榜与后期研究记录](b_board_records.md)：最终训练选择、候选选择、几何和数值路线；保留各自评测范围，不补造对称编号。
- [Oracle 分析](oracle_analysis.md)：特权信息收益、逼近路线、失败原因与证据边界。
- [研究范围与结论](research_findings.md)：工作范围、已排除的具体配置和仍未解决的问题。
- [证据目录](evidence/README.md)：指标摘录、来源哈希、结构身份修正和分类索引。

跨类别实验只有一个主分类，其他页面可以交叉引用。线上、本地 MOCK（模拟评测）和 Oracle（使用特权信息的参照）不合并排名，不同基线的增量不相加。
