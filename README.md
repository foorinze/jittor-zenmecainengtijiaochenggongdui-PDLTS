# PDLTS：Jittor 三维点云去噪竞赛开源仓库

本仓库提供第七届计图人工智能挑战赛赛道二的 A/B 榜最终方案、权重、训练推理代码及实验依据。任务输入为带噪三维点云，输出保持相同点数，最终推理不读取干净点云、真实网格或标签。仓库命名遵循赛题组要求：

```text
jittor-怎么才能提交成功队-PDLTS
```

## 成绩

| 榜单 | 名次 | 分数 |
|---|---:|---:|
| A 榜 | 第 5 名 | 84.10（CD 74.08 / P2S 94.13） |
| B 榜 | 第 8 名 | 81.05（CD 70.01 / P2S 92.10） |

## 最终方案

最终方案以 PD-LTS Light 为基础，在 Jittor 中采用仿射耦合层实现可逆变换，并用第一遍实际输出训练第二遍专训模型。B 榜推理流程为：

```text
带噪点云 -> 第一遍基础模型 -> 第二遍专训模型 -> 提交结果
```

第一遍模型在 B 榜训练数据上从零训练 100 轮，再通过 strict resume（严格续训，恢复模型、优化器和随机状态）训练到 ep149。第二遍 specialist（第二遍专训模型）从第一遍权重初始化，在固定 2000 个训练形状的第一遍输出上训练 50 轮。

A 榜最终方案还使用评分归一化损失微调和冻结几何方向场后处理；B 榜采用普通双向 Chamfer（倒角距离）损失，不使用这两项组件。两榜配置分别由对应评测确定，见 [A/B 榜算法改动](docs/a_to_b_changes.md)。

## 目录与入口

```text
a_board/       A 榜完整复现：模型、配置、脚本与两份权重
b_board/       B 榜完整复现：模型、配置、脚本、两份权重及结果来源
experiments/   分类实验、A 榜完整记录、B 榜与后期研究、指标证据
lessons/       按研究问题归纳的经验、未采用原因与适用边界
docs/          方法、两榜差异、评测规范、来源与发布范围
validation/    验证脚本与 reports/ 检查报告
manifests/     整个公开目录的文件哈希清单
LICENSES/      第三方许可证
```

| 阅读目的 | 入口 |
|---|---|
| 复现 A 榜 | [A 榜方案与复现](a_board/README.md) |
| 复现 B 榜 | [B 榜方案与复现](b_board/README.md) |
| 查实验设置、结果与证据 | [分类实验记录](experiments/README.md) |
| 查哪些做法有效、哪些条件下失败 | [实验经验](lessons/README.md) |
| 理解算法和评分 | [方法概览](docs/method_overview.md)、[评测规范](docs/evaluation_protocol.md) |
| 核查公开范围和文件 | [文档目录](docs/README.md)、[发布验证](docs/release_preparation.md) |

A/B 分别保留完整运行链，共用环境依赖和文档。数据集、预测数组和大型缓存不随仓库分发。

## A 榜快速复现

将测试数据放入 `a_board/dataset/test_noisy/`，安装根目录环境依赖后执行：

```bash
bash a_board/run_all.sh infer
```

该命令使用随包权重，执行两遍推理和冻结方向场后处理。数据结构、从零训练与各阶段配置见 [A 榜复现说明](a_board/README.md)。

## B 榜快速复现

环境建议：

- Ubuntu 22.04
- Python 3.9.25（已测环境）
- CUDA 12.x（云端完整验证使用 12.2）
- Jittor 1.3.10
- 24GB 显存 GPU

安装依赖：

```bash
python -m pip install -r requirements.txt
```

将官方 B 榜测试数据放到：

```text
b_board/dataset/test_noisy_b/shapenet/00000000/<sample_id>/noisy.npy
```

测试集应有 200 个样本。然后在仓库根目录执行：

```bash
bash b_board/reproduce_b_final.sh
```

脚本会检查数据数量、执行两遍推理、确认每遍输出完整，并生成 `result.zip`。

## 从原始训练数据重训

将官方 B 榜训练 mesh 放到：

```text
b_board/dataset/train_b/shapenet/00000000/<sample_id>/models/model_normalized.obj
```

然后执行：

```bash
cd b_board
bash scripts/experiments/b_final/train_b_final_pipeline.sh
```

`b_final` 目录包含最终 B 榜方案的训练脚本及配置。目录与配置对应关系见 [目录与配置名称](docs/public_naming_map.md)。

## 复现实验边界

随仓库权重可以复现 B 榜 81.05 对应的推理链。由于 2026-08-11 首次生成 B 榜训练对时使用了 Python `hash()` 且未固定 `PYTHONHASHSEED`，无法从原始 mesh 逐位重建当时的训练点云对；当前重训脚本已改用 SHA-256 派生稳定种子，能复现训练流程，但不能保证新权重与线上提交权重逐位一致。

## 方案选择依据

PyTorch 横评中，训练 20 轮的 PD-LTS Light、Heavy、StraightPCF 和 ScoreDenoise 的 mock20（20 样本本地模拟评测）分别为 82.68、57.89、70.20 和 43.97。Light 在已测配置中得分最高，记录训练耗时约为 Heavy 的三分之一。Jittor Heavy 的 RTX 4090 长训反复失败，后续 Heavy + EMD（地球移动距离）实验也未达到继续扩量要求。各方法的初始化和训练预算存在差异，现有结果没有确定 Heavy 的充分收敛表现。

Light 的 Jittor 改造主要减少数值求解器的移植和维护成本：仿射耦合直接求逆，使用常规自动求导。历史名义单调/仿射运行的结构身份存在疑点，不能用其相同分数证明结构等价；原始分数与核查边界见 [结构实验](experiments/architecture.md)。

基础模型、同模型重复推理和第二遍专训分别比较。A 榜同模型二遍在已测基座上有效，三遍回落；B 榜同模型二遍负向，但针对第一遍实际输出训练的专训模型有效。最终采用两遍专训，依据是对应数据和基座上的完整点云评测。详细设置与证据见 [实验结果与方案选择](experiments/overview.md)。

## 剩余误差与研究结果

后期研究检查了匹配蒸馏、位移学习、曲面恢复、密度校准和候选选择。Oracle（使用特权信息的理想参照）在 A 榜局部匹配实验中达到 99.51，同次 mock200 基线为 83.72；晚期 B 榜固定候选池的真值选择提高约 11.11 分。这些结果说明已测输出仍有改进空间，不计入线上成绩，也不是对所有方法求得的严格最优上界。

B 榜可部署分类器的 ROC-AUC（分类排序指标）为 0.670，直接全局选点仍下降约 10.81 分，覆盖误差显著增大。其他对照还发现，位移幅度校准不保证连续更新有效，原始几何误差改善不保证归一化评分改善。已有证据支持分别检查信息恢复、模型表示、集合选择和评分目标，尚未证明不可突破的信息限制。

工作范围、受实验条件限制的结论与后续检查方法见 [研究范围、对照结果与经验](experiments/research_findings.md)；各类特权参照与逼近方法见 [Oracle 分析](experiments/oracle_analysis.md)。公开目录提供指标摘录及来源哈希，历史诊断代码、缓存和检查点未全部提供，完整研究尚不能全部一键复跑。

## 文件来源与验证

代码来源及随仓库提供的内容见 [代码来源与发布范围](docs/release_scope.md)。

当前目录的文件清单、验证结果及检查命令见 [发布准备与验证](docs/release_preparation.md)。

## 许可与引用

本项目原创部分采用 [MIT 许可证](LICENSE)，第三方代码保留各自授权，详见
[第三方来源与许可](THIRD_PARTY_NOTICES.md)。软件引用信息见 [CITATION.cff](CITATION.cff)，
PD-LTS 基础论文引用见第三方声明。比赛数据不随包提供。
