# 代码来源与发布范围

本仓库以 B 榜最终代码提交中的 A/B 复现代码、权重和结果清单为基础整理。
A 榜快照与当时的正式代码检查材料交叉核对，实验结果补充自训练日志和评测文件。

## 随仓库提供的内容

| 内容 | 位置 |
|---|---|
| B 榜基础模型与第二遍专训权重 | `b_board/checkpoints/` |
| A 榜代码、配置、权重和运行入口 | `a_board/` |
| B 榜模型、训练与推理配置 | `b_board/starter_code/` |
| B 榜训练、缓存生成与推理脚本 | `b_board/scripts/` |
| B 榜最终推理入口 | `b_board/reproduce_b_final.sh` |
| 方法比较、消融和未采用实验 | [实验结果](../experiments/overview.md)、[A 榜实验记录](../experiments/a_board_atlas.md) |
| 后期研究与指标证据 | [研究范围与经验](../experiments/research_findings.md)、[Oracle 分析](../experiments/oracle_analysis.md)、[文档目录](README.md) |
| 按问题归类的实验经验 | [实验经验与路线复盘](../lessons/README.md) |
| 训练与推理复现说明 | [复现指南](../b_board/reproduction.md) |
| B 榜最终提交结果与权重的对应关系 | `b_board/results/` |
| 本地验证脚本 | `validation/` |

`pdlts_light` 是最终模型。Heavy、单调可逆层和 EMD 代码作为对照实现保留，
不用于最终 A/B 推理。未提供明确授权的 DCD 改写不在本仓库分发，
详见 [第三方来源与许可](../THIRD_PARTY_NOTICES.md)。

## 数据与历史记录

比赛数据、预测数组、完整训练日志和本地缓存不随仓库分发。
公开结果清单保留样本数、评测范围、权重哈希及来源摘要；
方法比较和消融证据包含原始指标、日志摘录与来源文件哈希。

86 条 A 榜实验记录按方法、数据、损失、结构、匹配和级联分组。
线上成绩、本地模拟评测和使用干净真值的理想参照分别报告。
研究工作区的报告和源文件数量用于说明记录范围，不等于公开文件数或独立实验次数。
完整历史诊断程序、几何求解器和检查点未全部分发；公开研究材料提供主要指标与结论依据，
尚不具备全部历史实验一键复跑条件。

## 当前状态

公开内容以 Git 目录形式维护。文件与验证状态见
[发布准备与验证](release_preparation.md)，命名和目录对应关系见
[目录与配置名称](public_naming_map.md)。
