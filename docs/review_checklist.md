# 文件复核与验证状态

## 入口与实现

| 范围 | 核对内容 |
|---|---|
| 根 README、复现指南、A/B 差异说明 | 环境、工作目录、数据路径、训练与推理命令 |
| `requirements.txt`、`environment.yaml` | Python 3.9.25 与依赖版本；Linux pip 解析通过 |
| `b_board/reproduce_b_final.sh` | B 榜两遍推理、样本完整性及结果打包 |
| `a_board/run_all.sh` | A 榜完整训练和仅推理两种入口；仅推理不要求训练集 |
| `b_board/scripts/experiments/b_final/` | 数据生成、基础训练、严格续训、缓存和第二遍专训 |
| A/B `evaluate_mock.py` | 评分、归一化、缺失输入和无效预测处理 |
| `pdlts_light` | 模型、损失、整云拼接、训练状态与输出清单 |
| `pdlts_heavy`、`imonotone_light*`、`emd_jittor` | 保留为对照实现，未用于最终提交 |
| 许可证和第三方声明 | 原创 MIT 与上游授权分别列明；未明确授权的 DCD 改写不分发 |
| 权重与结果清单 | 四份 A/B 权重哈希与来源清单一致 |

## 实验记录

| 文件 | 内容 |
|---|---|
| [实验结果与方案选择](../experiments/overview.md) | 方法选择、Light/Heavy、Jittor 改造、损失和结构消融、两遍去噪及最终配置 |
| [A 榜实验记录](../experiments/a_board_atlas.md) | 86 条实验的设置、结果和采用情况 |
| [方法比较证据](../experiments/evidence/method_comparison_evidence.json) | 11 个已完成评测节点的分数、日志和哈希 |
| [Jittor 迁移与消融证据](../experiments/evidence/jittor_adaptation_evidence.json) | Heavy 失败摘录、结构和损失消融指标及来源 |
| [评测与归一化](evaluation_protocol.md) | A/B 本地评测命令、坐标变换和失败实验 |
| [研究范围与经验](../experiments/research_findings.md)、[研究指标](../experiments/evidence/research_evidence.json) | 工作范围统计、机制对照、晚期 B 榜候选研究与结论范围 |
| [Oracle 分析](../experiments/oracle_analysis.md)、[Oracle 指标](../experiments/evidence/oracle_evidence.json) | 特权参照、可部署逼近方法与未解决的问题 |

根 README 汇总最终方案和复现入口，[文档目录](README.md)按方案与复现、研究结果、指标证据、发布验证组织材料。研究数字保留评测集合与比较对象，线上分数、本地增量和特权诊断不混算。

## 验证结果

| 检查 | 状态 |
|---|---|
| 语法、文档链接、文件扫描及权重哈希 | [静态检查报告](../validation/reports/release_static_validation.json) |
| A/B 合成评测样例 | NumPy 与 pcu-BVH 后端各通过 22 个用例 |
| A/B 命令入口 | 7 个用例通过 |
| A/B 模型默认损失与梯度 | Jittor CPU 检查通过，排除的 DCD 分支不能启用 |
| 云端 Jittor GPU | [历史验证记录](../validation/reports/cloud_jittor_validation_20260905.md)，含 B 榜 200 样本双遍推理 |
| 发布文件 | 以 Git 目录和文件清单维护，按路径与哈希核对公开内容 |

详细命令和报告见 [发布准备与验证](release_preparation.md)。
新 conda 环境完整创建、真实训练 OBJ 分支及从零训练未在这轮发布验证中执行；
历史训练点云对不能逐位重建的限制见 [复现指南](../b_board/reproduction.md)。
