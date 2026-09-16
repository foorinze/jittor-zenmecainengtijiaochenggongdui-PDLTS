# A 榜实验记录

本文汇总 A 榜研发中的方法比较、对照实验、诊断结果和采用情况。数据来源包括实验报告、执行规格、评测摘要、训练记录和提交记录。

主要结果见 [实验结果与方案选择](overview.md)，各项实验在下表中使用独立 ID 标识。

后期真值梯度更新、曲面修正、轨迹监督和候选选择的对照，另见 [Oracle 实验与可部署方案的差距](oracle_analysis.md)。该文补充各类 oracle 的收益、实际方案未能达到的原因及结论边界；[指标摘录](evidence/oracle_evidence.json)保留原始字段与来源哈希。

## 1. 记录范围

本文记录竞赛 A 榜实验。`mock20`、`mock200` 分别使用 20、200 个本地样本，
`online` 为官方线上成绩。`oracle`（理想上界）实验读取 clean（干净点云）或其派生标签，
记录使用真值时可达到的结果，与部署时只读取带噪输入的模型分别比较。

仓库提供 A 榜最终代码快照和 B 榜最终复现链。历史实验的全部代码、环境和检查点尚未随包提供，具体范围见第 15 节。

本文统一使用以下术语：`specialist`（第二遍专训模型）、`coverage`（覆盖率）、`assignment`（匹配分配）、`oracle`（读取特权信息得到的理想上界）、`full-cloud`（完整点云评测）、`matched control`（只改变一个变量的匹配对照）。

## 2. 记录类型

| 标记 | 内容 | 解释范围 |
|---|---|---|
| `SCORE` | 走完整 prediction -> evaluate_mock -> metrics 链，或有线上记录 | 支持该配置在指定评测范围内的分数结论 |
| `DIAG` | 配对诊断、coverage、位移统计或类别拆分 | 解释分数或定位瓶颈，不单独支持提交 |
| `GATE` | 代码、数值、梯度、shape 和恢复状态检查 | 支持“能运行/实现通过”，不等于有分数提升 |
| `ORACLE` | 使用 clean、mesh、assignment 标签或特权目标 | 支持上界或可达性判断，不等于可部署 |
| `BLOCKED` | 环境、显存、算子或迁移契约阻塞 | 该配置未完成运行，不用于比较方法的充分收敛表现 |
| `NEGATIVE` | 评测链有效且结果不如对照 | 可以关闭冻结配置，结论限于已测配置 |
| `WEAK_POSITIVE` | 有小幅正向或只有局部指标正向 | 需要 matched control（匹配对照）或更大范围复核，尚不足以采用 |

## 3. 基线与提交结果

以下结果的训练配置和评测范围不同，不作跨行排名。

| 公开锚点 | 评测范围 | Final | CD | P2S | 用途 |
|---|---|---:|---:|---:|---|
| 初始流程 smoke | `mock200`，仅少量有效预测，其余按 0 计 | 0.00 | 0.00 | 0.00 | 评测和结果打包可运行 |
| Light 第一份长训 | `mock200` | 82.58 | 69.29 | 91.02 | 第一份可提交基线 |
| Light 长训增强 | `mock200` | 83.72 | 70.72 | 91.96 | 证明长训有收益，但仍有平台 |
| 全覆盖 Light 基座 | `mock200` | 84.06 | 74.28 | 93.83 | 二遍路线的父模型 |
| 父子二遍融合 | `mock200` | 84.91 | 74.80 | 95.01 | 证明同点位二遍融合有互补 |
| 二遍专训历史最佳 | `mock200` | 86.45 | 77.00 | 95.90 | A 榜后期候选基座的本地锚点 |
| A 榜最终提交快照 | `online` | 84.10 | 74.08 | 94.13 | 公开仓库 README 声明的最终 A 榜成绩 |

补充：内部记录里还存在 `84.09` 的另一历史候选线上记录。它不是本文的 A 榜最终成绩，公开版不将两个提交节点合并；最终 A 榜以正式成绩 `84.10` 为准。

## 4. 全部研发阶段概览

实验按方法比较、数据、损失、架构、匹配和级联等类别分组。

| 公开阶段 | 核心问题 | 结果 |
|---|---|---|
| `A-SELECT` 方法比较 | 四种 PyTorch 配置的分数及训练成本 | Light 在已测配置中得分最高，本次训练耗时低于 Heavy，采用 Light |
| `A-FND` 基础实现 | Jittor 版 Light 能否完整训练、推理、评测、打包 | 通过，得到第一份可提交基线 |
| `A-DATA` 数据调度 | 训练集覆盖、shard 轮转是不是平台根因 | 不是主因 |
| `A-LOSS` 损失和后处理 | 改 loss、局部间距、方向、密度是否能破位移平台 | 大多失败或只有弱信号 |
| `A-ANCHOR` 监督锚点 | 完整性权重和配对目标能否修复保守位移 | 局部正向未稳定转成分数 |
| `A-ARCH` 架构和容量 | Heavy、图网络、显式方向场、完整可逆结构是否更强 | 部分结构有上界，当前可部署转化失败 |
| `A-EMD` 完整结构路线 | 官方 EMD + iMonotone 能否在 Jittor 从零兑现 | 详见 `A-ARCH-05/06`；工程可行但 full15k 配方实例失败 |
| `A-ASSIGN` 指派和传输 | 一对一匹配、Hungarian、Sinkhorn 能否改善 y2x 覆盖 | oracle 强，学习器连续失败 |
| `A-TRANSPORT` 候选池、传输和选择器 | 扩点、many-to-few、selector 能否修 coverage | 候选池上界不足，训练选择器不稳定 |
| `A-BRIDGE` oracle 到可部署 | clean 派生的理想移动是否可由 noisy/base 观察到 | 可观测信号弱，官方分数转化失败 |
| `A-CASCADE` 二遍链路 | 第一遍输出能否成为第二遍更合适的输入分布 | 成功，成为最终方案来源 |

### 4.1 `A-SELECT`：代表方法横评与 Light/Heavy 取舍

横评训练均在 PyTorch 中进行，评分范围为 A 榜研发 `mock20`。原始评测日志及 SHA-256 见 [横评证据摘录](evidence/method_comparison_evidence.json)，它包含 11 个观测节点的完整标准输出。epoch（轮次）从 0 编号，ep19 对应训练 20 轮。

| ID | 研究问题 | 实际方法和预算 | mock20 结果 | 结果与采用依据 |
|---|---|---|---|---|
| `A-SELECT-01` | PD-LTS Light 是否能在本题数据上形成有效基座 | 随机初始化；固定点云对；官方单段可逆结构与有效 EMD 损失 | 20 轮 `82.68 / CD 74.17 / P2S 91.18`；100 轮 `84.43 / 76.33 / 92.52` | `SCORE`；基座有效且延长训练仍有收益 |
| `A-SELECT-02` | PD-LTS Heavy 是否更值得投入 | 随机初始化；三阶段级联与分阶段 EMD；经验噪声尺度构造中间目标 | 5 轮 `45.87`；20 轮 `57.89 / 48.91 / 66.88` | `NEGATIVE`；当前配方不如 Light，不据此否定所有 Heavy 实现 |
| `A-SELECT-03` | StraightPCF 预训练微调是否适合本题 | 预训练 CVM 与随机初始化距离头，全网络微调；在线加噪 | 5/20/40 轮分别 `71.86/70.20/71.58` | `SCORE`；已测微调节点没有持续提升，不能标为从零训练或直接零样本推理 |
| `A-SELECT-04` | ScoreDenoise 的分数场学习是否能产生更强去噪 | 随机初始化；在线加噪；DSM（去噪分数匹配）；固定步长迭代推理 | 5/20/50/100 轮分别 `0.00/43.97/62.01/53.55` | `SCORE`；保留学习进展和回落，不把 62.01 当方法能力上限 |
| `A-SELECT-05` | ScoreDenoise 回落能否通过推理步长选择修复 | 训练集少量样本的孤立 patch（局部块）探针、完整拼接流程探针 | 孤立探针在 20 轮节点为 `22.33`；完整流程探针在 20/50/100 轮为 `37.43/33.71/41.42`，均低于对应固定步长结果 | `DIAG` + `NEGATIVE`；两次修复均未形成可靠整体收益，步长影响仍是解释边界 |
| `A-SELECT-06` | 家族内部选择 Light 是否有成本与分数依据 | Light/Heavy 同为 20 轮；同一训练点云对来源；比较整体配方 | Light/Heavy 为 `82.68/57.89`；记录训练耗时约 `5.9/17.8` 小时 | `SCORE` + `DIAG`；支持本次工程取舍，阶段结构、目标和学习率同时变化，非纯容量消融 |

前四项评分日志均记录 20/20 有效预测、无缺失、无点数不匹配，以及基于 `norm.json` 的 pcu-BVH 精确 P2S。Light 20 轮采用精确重评结果；原先缺少 pcu 的回退结果仅用于解释评测故障，不作为方法成绩。

**采用配置：** PD-LTS Light。它在四种已测配置中分数最高；与 Heavy 同为 20 轮时，记录训练耗时更短。Jittor Heavy 的长训稳定性及后续 Heavy + EMD 分数实验还分别见 `A-LOSS-06`、`A-ARCH-11`。初始化、加噪方式、batch（批大小）、实际优化步数、设备和追加预算不统一，比较结果限于本次配置，未确定 Heavy 的后程表现。

第 5 项来自步长修复执行记录，单独标为诊断与负例；证据 JSON 当前提供的是固定步长主评测节点，不声称包含全部诊断原始产物。

## 5. `A-FND`：训练与评测基线

| ID | 假设 | 实际方法和对照 | 结果 | 结果类型 | 经验 |
|---|---|---|---|---|---|
| `A-FND-01` | 检查迁移、评测和结果打包 | Jittor Light；MLGC、AffineCoupling、ActNorm、FPS+KNN stitching；97 个代码/单测；clean/noisy 黄金样例 | `clean=100.00`，`noisy=0.00`；shape、缺失预测、finite 检查通过 | `GATE` | 评测器和输出检查通过 |
| `A-FND-02` | 小规模 smoke 学不到不代表模型无效 | 16 个样本、1 epoch smoke；同时跑 20 样本子集和 200 样本全口径 | 两种口径都为 0；预测比 noisy 更差 | `GATE` | 该 0 分来自少量样本和单轮训练，不足以评价模型能力 |
| `A-FND-03` | Light 长训可以超过原基线 | 10k 训练规模、约 50 epoch、固定推理规则 | `mock200=82.58`，`online=80.15` | `SCORE` | 仿射耦合与替代损失得到可用的长训基线 |
| `A-FND-04` | 只延长训练可以继续提升 | 继续到约 100 epoch，按 checkpoint 做 ladder | `mock200=83.72`，`online=81.34`；后期在约 83 分附近摆动 | `WEAK_POSITIVE` | 长训有收益，但“再加 epoch”不是无限增益轴 |
| `A-FND-05` | 评测异常可能来自输出覆盖不全 | 扫 `seed_k`，单独测 FPS+KNN coverage；比较 L1/L2/L3 回填 | `seed_k=3` 初始覆盖不稳定；`seed_k=12` 在真实 200 样本上达到 zero-miss | `GATE` | 覆盖统计、缺点回填和评测分数分别记录 |
| `A-FND-06` | 训练期指标能解释线上落差 | paired displacement 诊断和 scale sweep | `disp_cos` 约 0.37，`disp_scale` 约 0.5，`paired_L2_ratio` 约 0.85；缩放后没有破局 | `DIAG` | 模型学成了保守去噪，主要缺口在 y2x 覆盖半边 |

### 5.1 位移诊断

基线的配对位移统计如下：

```text
输入 noisy -> 预测 displacement
                 方向相关性约 0.36~0.37
                 位移幅度约为目标的一半
                 完整点云的 clean-to-pred 覆盖仍有缺口
```

相关对照包括：

- 改变监督的方向和幅度；
- 改变点集合的匹配/覆盖关系；
- 使用第一遍输出训练第二遍模型。

## 6. `A-LOSS`：损失、后处理和局部几何路线

| ID | 假设 | 方法 | 结果 | 结果类型 | 关闭原因或保留价值 |
|---|---|---|---|---|---|
| `A-LOSS-01` | LOP/WLOP 后处理可以通过局部间距改善覆盖 | 对已有预测做 48 组左右的 LOP/WLOP 参数扫描 | 一致出现 P2S 上升、CD 下降，Final 没有超过基线 | `NEGATIVE` | P2S 与 CD 的变化相反，未采用该后处理 |
| `A-LOSS-02` | 显式方向监督可以解除隐式 displacement 塌缩 | `dir + scale` 双头、cosine + magnitude loss；1k/20ep 先看结构指标 | 训练 patch 内方向指标变好，但 mock20 整云 `disp_cos=0.3505`，匹配对照 `0.3433`；Final 约 74.43，对照约 75.11 | `NEGATIVE` | 结构信号没有转成分数，且预算不等同长训基线 |
| `A-LOSS-03` | fixed-index 配对本身可能是方向平台根因 | 对同一预测比较 fixed、nearest-neighbor、OT 三种配对 | fixed cosine 约 0.36，NN cosine 约 0.75；但不改变可部署预测 | `DIAG` | 说明标签配对有错配，但 oracle 配对不能直接当训练 target |
| `A-LOSS-04` | 重新截断 FBM 或做 NN retarget 可以修覆盖 | 试验 FBM cut、NN retarget、局部重定位；先做 dry-run，再做小预算 | 只有弱 coverage/配对信号，没有稳定官方分数正向 | `WEAK_POSITIVE` | 作为后续 assignment 研究的诊断材料，不进入提交链 |
| `A-LOSS-05` | density-aware Chamfer / DCD 能直接改善重复和稀疏 | 分开核对 `dcd_like` 与 official DCD；固定其他训练条件 | `dcd_like` 不能作为 official DCD 证据；official DCD 只有 weak positive，机制未激活 | `WEAK_POSITIVE` | 早期报告曾把两个实现混称，后续已明确拆开，未采用 |
| `A-LOSS-06` | 更大 Heavy 网络能突破 Light 平台 | 早期 Jittor Heavy 近似实现；梯度与数值修复、direct-10k 和短周期续训 | 1k/b4/50 轮可完成，mock20 为 55.02；direct-10k b8/b4 均发生 CUDA 非法地址访问；RTX 4090 的 micro80–89 段三次未完成，末次在求根路径报矩阵乘法执行失败；最高有效 mock20 为 micro79 的 72.82 | `BLOCKED` | 未完成计划内长训；micro79 不是完整 79 轮，无法判断充分收敛或后程优势；底层 CUDA 根因未确定 |
| `A-LOSS-07` | 预测点之间的 repulsion 可以减少重复 | training-time pred-pred repulsion；比较强度和 P2S 安全 | 强配置触发 P2S unsafe，较弱配置仍比基线低约 3.5 Final、P2S 低约 4.46 | `NEGATIVE` | 只调 spacing 会破坏 anchor，dry-run 梯度正交不足以保证长训安全 |
| `A-LOSS-08` | sliced Wasserstein 可以提供更好的集合约束 | vanilla global、per-sample、不同 K 和权重；与 C5/Hungarian 对照 | vanilla `Final=74.30` vs 基线 `74.75`；长训 best `mock200=83.38` vs 基线 `83.72` | `WEAK_POSITIVE` | 局部 coverage 有变化，但没有突破分数或位移机制平台 |
| `A-LOSS-09` | Hungarian matched L2 是稳定的第三项 loss | 先做 full N=1024 timing、梯度夹角和无 NaN 检查，再做 1k/20ep、继续训练 | dry-run 全通过；最佳延续点 `Final=76.01`，coverage `0.5855`，但未达到扩大训练的阈值 | `WEAK_POSITIVE` | 实现保留为研究组件，未扩大到 10k 训练 |
| `A-LOSS-10` | local spacing residual 是独立误差来源，专用 spacing loss 可以修复 | 对 `k=4/8/16` 的局部间距尾部做诊断；只有诊断，不直接训练 | `k=4` 仍有 residual，但 `k=8/16` 信号快速衰减；pred 相对 noisy 已明显改善，不能证明独立 loss 有效 | `DIAG` | 未继续训练 spacing loss；现有诊断受样本规模和切分影响 |
| `A-LOSS-11` | dense surface auxiliary 可以把预测点更稳定地贴到表面 | 只增加 pred-to-dense-clean 的单向贴面项，权重 `0.03/0.05/0.10`，matched control 保持相同采样 | 最好档 `Final=82.71 / CD=74.11 / P2S=91.30`，control 为 `82.75 / 74.17 / 91.33`；coverage 无改善，高权重轻微变差 | `NEGATIVE` | x2y 已接近饱和；保留 dense 数据契约和代码基础设施，不把它解释成 assignment 路线失败 |
| `A-LOSS-12` | 单纯增大 L2 或增加 noisy seed augment 可以解除 displacement 塌缩 | L2 权重 `0.1/0.5/1.0` 与 noisy-seed augment 组合；保持其余训练设置 | `disp_scale` 仍约 `0.37~0.48`，`paired_L2_ratio` 约 `0.86~0.87`；L2 越大反而更保守，未形成方向改善 | `NEGATIVE` | 未继续扩大这组损失与增强配置的训练规模 |

### 6.1 未采用原因

Heavy 早期实现修复后可以学习，但长训因 CUDA 错误中断；现有日志不能将这些错误直接归因为显存不足。LOP/WLOP 和排斥损失完成评测后损害了 CD 或 P2S。Hungarian、Wasserstein 和 official DCD 的部分局部指标改善，但 Final 未稳定超过对照。这些配置均未进入最终方案；未完成的 Heavy 长训不记作收敛后的模型负结果。

## 7. `A-DATA`：训练覆盖和数据调度

| ID | 假设 | 方法和预算 | 结果 | 结果类型 |
|---|---|---|---|---|
| `A-DATA-01` | 旧训练每 epoch 只看约 10k 样本，覆盖不足导致平台 | full15k，每 epoch 遍历完整训练清单 | `mock20` ep99 `Final=83.26`；`disp_cos` 仍在 0.35~0.365，`disp_scale` 仍未稳定超过 0.5 | `NEGATIVE` |
| `A-DATA-02` | 固定 shard 轮转能改善随机采样噪声 | 全局打散后切 16 个约 1k shard，每个 macro epoch 轮转 | ep39/49 比同节点小幅高约 `+0.30/+0.36`，但机制指标没有移动；后段还混入 model-only resume 和 Adam 重置 | `WEAK_POSITIVE` |
| `A-DATA-03` | 训练顺序变化是主要提升来源 | 对 full15k 与 fixed-shard 逐节点比较 CD/P2S/位移指标 | full coverage 和 shard rotation 都只能带来弱分数变化，平台仍在 | `NEGATIVE` |
| `A-DATA-04` | 多喂数据给 specialist 可以继续明显涨分 | 固定算力比较约 2,000 与约 3,750 shape 的第二遍训练缓存 | Final 只多约 `0.02`，CD 基本不动 | `NEGATIVE` |

全量训练与分片轮转未消除位移平台。全量训练模型保留为二遍专训的基础模型；分片轮转未成为独立方案。

## 8. `A-ANCHOR`：C5、FCD 和 anchor 约束拆分

| ID | 假设 | 方法 | 结果 | 结果类型 |
|---|---|---|---|---|
| `A-ANCHOR-01` | C5 的 coverage 正信号在更长训练中会兑现 | C5/Hungarian 长训 baseline | coverage 有早期和局部正向，但 Final/CD/P2S 没有稳定超过对照 | `WEAK_POSITIVE` |
| `A-ANCHOR-02` | 加大全局 clean-to-pred 完整性权重能改善 CD | C5 + FCD static，固定 `beta=2` | ep19 `Final=79.35`，对照 `80.69`；P2S 低约 1.95 | `NEGATIVE` |
| `A-ANCHOR-03` | FCD 的问题只是权重退火太慢 | C5 + FCD linear/stair/exp schedule | Static 和 Linear 都不能在 CD/P2S/Final 上超过 S0；该轴停止 | `NEGATIVE` |
| `A-ANCHOR-04` | fixed-index L2 anchor 错配，改成 NN target 会更好 | C5 + `pred_nn_clean` target | ep19 一度 `Final=81.16`，但 ep49 对照 `82.26`、该分支 `82.24`；早期收益蒸发 | `NEGATIVE` |
| `A-ANCHOR-05` | 只降低 suspect pair 权重可以保住几何 | C5 + fixed-index suspect downweight | ep19 `Final=80.91`，ep49 `81.92`，低于对照；诊断更干净但力道不足 | `NEGATIVE` |

上述完整性权重与配对目标配置在最终评测节点未超过各自对照，均未采用。

## 9. `A-TRANSPORT`：覆盖、匹配和候选池

| ID | 假设 | 方法 | 结果 | 结果类型 | 关闭边界 |
|---|---|---|---|---|---|
| `A-TRANSPORT-01` | 先改变 target 构造，再用 Chamfer 主导训练即可改善 coverage | target 由 clean-nearest/ noisy-seed 等方式构造；Chamfer 权重 1.0 | coverage 前期上升，约 ep12 collapse | `NEGATIVE` | target 有早期信号，但 Chamfer 不能独立承担 unpaired 场景主 loss |
| `A-TRANSPORT-02` | Hungarian-main 可以阻止 collapse 并提高 coverage | target + Hungarian + Chamfer 0.1，1k 与 10k 对照 | 10k/50ep coverage 约 `0.623`、无 collapse，但 `mock200 Final=75.35`，远低于 83.72 基线 | `NEGATIVE` | set coverage optimization 没有转成 paired displacement score |
| `A-TRANSPORT-03` | 也许只是 stitching/merge 把好输出弄坏 | random merge、pooled output、standard stitching 的 oracle 对照 | mapping oracle 相对标准拼接提升小于 1% | `NEGATIVE` | 主因是输出质量和修正力度，不是最后一层 merge |
| `A-TRANSPORT-04` | Chamfer 权重调低可以避免 collapse | 权重 0.5、0.3；两种 seed | 0.5 两次均在约 ep12 collapse；0.3 不塌但 `mock20 Final=66.21`，低于同尺度 69.08 | `NEGATIVE` | 当前 1k 预算不能提供有效正向窗口 |
| `A-TRANSPORT-05` | latent consistency 可以把不同点对齐到稳定潜空间 | 预测 latent 与 target latent 的一致性探针 | `predict_z` 没有朝 target 子空间收敛；context drift 很大 | `NEGATIVE` | 关闭当前 latent 目标，不再追加同类调参 |
| `A-TRANSPORT-06` | 缩小 matching pool 或加 post-FBM residual head 可以救回训练 | subpatch Hungarian、post-FBM residual head | 两条都在小规模附近 collapse，coverage 约 0.57~0.58 | `NEGATIVE` | 局部化 assignment 或再加一个 head 没有改变训练动力学 |
| `A-TRANSPORT-07` | 纯 density transport 可以修 y2x | CD gap decomposition、uniformization、density proxy | 诊断确认 y2x 是主要弱半边；uniform 化几乎无收益，oracle 上界也不足 | `DIAG` + `NEGATIVE` | 未继续单独训练密度均匀化模型 |
| `A-TRANSPORT-08` | patch-local targeted transport 训练可以搬到 full-cloud | patch-local target、full-cloud replay、global density/source proxy | dry-run 可过，真实训练后 source 被推动但几乎不朝 full-cloud target 移动 | `NEGATIVE` | patch-local 正向不能替代 full-cloud score 证据 |
| `A-TRANSPORT-09` | keep-weight 能从 patch 监督变成全云选点 | soft per-point weight / keep-head，多个 matched control | 学到的高权重点在 stitching 后多为重叠冗余；真实参与选点后反而伤 coverage | `NEGATIVE` | 关闭当前 keep-weight，不启动重写 stitching 的 V2 |
| `A-TRANSPORT-10` | 插值/扩点候选池 + many-to-few 选择器能修 coverage | clean-guided facility oracle，固定点数选择 | 公平主臂 `Final=84.96 / CD=76.39 / P2S=93.52` | `ORACLE` + `NEGATIVE` | 机制有净杠杆，但候选池物理上限不足以支撑高目标 |
| `A-TRANSPORT-11` | 只提升候选池质量就能显著抬高上界 | PCA tangent、density-biased、ensemble 候选池 | 最强公平扩池约 `Final=84.04`；clean leakage 上界约 `88.68`，仍不足目标线 | `NEGATIVE` | 当前候选池族关闭，不能把 clean leakage 上界当部署预期 |
| `A-TRANSPORT-12` | learned replacement selector 可以学会 oracle 选择 | selector soft training + hard deployment 对照 | 有梯度，但 soft objective 与 hard deploy 脱节，mock20 收益约 `+0.01` | `NEGATIVE` | 关闭当前 selector，保留静态安全地板作为诊断 |
| `A-TRANSPORT-13` | 经典 mesh reconstruction 可以填补预测点云覆盖缺口 | Poisson、BPA、以及与 base 按比例混合的 hybrid | Poisson trim `17.79/47.46`；BPA full replace `76.55`；最好 hybrid `82.88`，仍低于 base `83.28`，BPA 比例越高越差 | `NEGATIVE` | Poisson 引入 phantom geometry，BPA 继承缺口并增加边缘 artifact；关闭经典重建线 |
| `A-TRANSPORT-14` | per-sample、per-class 或 per-region selector 的 oracle 上界足以支撑学习 | 扩大候选池后做 clean-guided oracle 选择，禁止把 clean 送入部署 | per-sample `84.71`、per-class `84.29`、region `84.53/84.63`，均低于预设 `85.02` gate | `ORACLE` + `NEGATIVE` | 候选之间互补性不够，连 oracle 选择都不到门槛，停止该选择器实验 |
| `A-TRANSPORT-15` | density/residual proxy 可以同时找到应删除的重复 source 和应补的 coverage source | 在 full-cloud 上分开审计 density、residual、noisy distance 与 rank-AND 组合 | 最好 duplicate precision `0.5326 < 0.60`；rank-AND 为 `0.4547`、oracle lift `0.88`；密度和高残差在当前可见特征中是两类点 | `NEGATIVE` + `DIAG` | 已测 z-score 与特征交集未提供足够的区分能力 |
| `A-TRANSPORT-16` | target-only y2x one-step 可以在不训练的情况下改善 full-cloud coverage | clean-truth 与 deploy-proxy 两条 target-only full-cloud dry-run | clean-truth `dy2x=+4.695e-06`、`dx2y=+1.434e-05`；200 个样本中 143/195 个变差；deploy proxy 下 198/199 个变差 | `NEGATIVE` | 目标构造没有把 source 选对，且同时伤害 x2y 与 y2x；关闭 target-only/W 线 |

## 10. `A-ASSIGN`：匹配上界与目标蒸馏

| ID | 假设 | 方法 | 结果 | 结果类型 |
|---|---|---|---|---|
| `A-ASSIGN-01` | 固定 50k 点仍有很大的 one-to-one CD 空间 | 使用 clean 构造局部 one-to-one assignment oracle | mock20 局部 oracle `CD=99.74`，assignment coverage `0.998479`；但 P2S 是 snap-to-clean artifact | `ORACLE` |
| `A-ASSIGN-02` | 离线 hard-LSAP residual target 可以蒸馏给 refiner | assignment cache fullcheck、再做 train100 小试 | cache 逐点一致；3 epoch pilot `Final=82.99`，低于 identity `83.01`；overfit 仍接近 identity | `GATE` + `NEGATIVE` |
| `A-ASSIGN-03` | 在线 Sinkhorn 比离线 hard target 更容易学 | soft OT/Sinkhorn feasibility、overfit-1、full-cloud replay | loss 能下降，但 replay 方向有害；regularized shot 的 `cd_proxy_improvement=-0.000123784` | `NEGATIVE` |
| `A-ASSIGN-04` | 只加输入端 normal 特征即可修方向 | single-scale PCA normal、多尺度 normal v2 | overfit gate 连续未通过；movement、duplicate 和 CD proxy 没有同时满足 | `NEGATIVE` |
| `A-ASSIGN-05` | 图通信能修复 pointwise refiner 的 collapse | EdgeConv-lite / neighborhood message passing | 梯度链健康，但 `duplicate_proxy=0.45856`，基线约 `0.36118`；早停，`cd_proxy` 反向 | `NEGATIVE` |
| `A-ASSIGN-06` | topology regularization 能让 hard assignment 可部署 | free-delta + local order/volume 约束、RBF/Wendland field | oracle field 有空间；learnable field 无法学会高频 target，未获分数授权 | `NEGATIVE` |
| `A-ASSIGN-07` | Transformer 只要容量够就能学高频 assignment field | local MLP 与 Transformer sanity 对照 | Transformer 提升低频 level0 cosine，但 level1 高频 target 仍约 0.44，realized delta gate 仍失败 | `NEGATIVE` |
| `A-ASSIGN-08` | 近似 assignment 的忠实度不足是训练失败的唯一原因 | local patch kNN + vote aggregation；只读 clean 的 fidelity oracle | `eta=0.20` 达 `87.46 / CD=80.80 / P2S=94.11`；`eta=0.50` 达 `93.90 / 90.25 / 97.54`，bijection 约 `84%` | `ORACLE` |
| `A-ASSIGN-09` | 将高 fidelity assignment 固化成训练 target 就能泛化 | NN greedy 与 local patch assignment 的 warm-start train-small 对照 | 50 epoch 后分别约 `81.30` 与 `81.29`，都低于 identity `83.01`；assignment 质量差异被训练动力学抹平 | `NEGATIVE` |
| `A-ASSIGN-10` | clean-oracle 的 assignment/transport 上界足以代表可提交上限 | slot-preserving oracle transport；对 duplicate 点做多种理想搬运策略，并做 mock20/mock200 复评 | mock20 `CD=89.95`，mock200 `CD=88.96`，但搬运点直接落到 clean 附近，P2S 同步出现 snap artifact | `ORACLE` |

### 10.1 蒸馏结果

两种匹配目标训练 50 轮后的 Final 约为 81.30 和 81.29，低于原预测的 83.01，未采用。理想匹配使用干净点云，单独列为上界实验；高质量匹配目标在本次训练中未产生对应的模型收益。

## 11. `A-BRIDGE`：部署可见特征与移动预测

| ID | 假设 | 方法 | 结果 | 结果类型 |
|---|---|---|---|---|
| `A-BRIDGE-01` | clean-oracle 的移动标签可以由 noisy/base 可见特征预测 | shape-heldout、GroupKFold，特征仅来自 noisy/base 和图统计 | top-move AUC 约 `0.6144`，幅度 R2 约 `0.1513`，方向 cosine 约 `0.084` | `WEAK_POSITIVE` |
| `A-BRIDGE-02` | 只用可见特征做保守 gate 就能得到真实收益 | forward/reverse displacement、local repulsion 等零训练规则 | 最佳相对 identity 约 `+0.12`，低于预设 `+0.20` 阈值；其他规则对 random gate 反而更差 | `NEGATIVE` |
| `A-BRIDGE-03` | oracle gate 本身没有价值，所有收益都来自随便移动 | identity、random gate、oracle gate ceiling 五臂消融 | oracle gate 相对 identity 有稳定正向，但其中约 69% 来自 oracle 级移动本身，真实 gate 只占约 31% | `ORACLE` + `DIAG` |
| `A-BRIDGE-04` | 预测期局部投影可以利用现有表面 | PCA / SDF / Neural Pull / surface projection | oracle 法向有明显 headroom，但预测法向/落点误差无法兑现；部分修正版甚至 Final 大幅下降 | `NEGATIVE` |
| `A-BRIDGE-05` | 外部方法横评的旧交叉索引 | 完整内容见 `A-SELECT` 选型组；原行混写随机初始化与预训练微调，现已纠正 | ScoreDenoise 包含 50 轮 `62.01` 与 100 轮 `53.55`；Light 包含 100 轮 `84.43` | 仅保留检索入口，不作为额外实验或“预训练直接迁移”的负例 |
| `A-BRIDGE-06` | NN pull/refiner 的 oracle 移动可以被可部署模型学会 | 单 shape overfit、train-small、可见特征 rescue、KNN memory transfer | overfit 可到 `90.28`；train-small 只有 `83.03`、KNN transfer `83.04`，最高仅约 `+0.03`，低于 `83.30` gate | `ORACLE` + `NEGATIVE` |

本组预测输入限于 noisy、基础预测和由其计算的特征。读取 clean、mesh 或 normal 真值的结果标为 oracle，不计入可部署模型成绩。

> 历史命名为 iMonotone 的 Jittor 20/100 轮运行，配置声明与训练参数清单存在冲突，当前包装器也未转发结构选择键。因此保留原始分数，但暂停把它们作为已认证的结构消融；不能用两组均为 83.26 证明结构相当。见 [结构身份核查](evidence/structure_identity_review.json)。

## 12. `A-ARCH`：完整 INN、EMD 和 Jittor 迁移

本组分别记录可逆层替换、前向数值对齐、损失替换和 Jittor 从零训练结果。

| ID | 假设 | 方法 | 结果 | 结果类型 |
|---|---|---|---|---|
| `A-ARCH-01` | 官方完整 Light 架构比降级 AffineCoupling 更强 | PyTorch 中锁定 MLGC、数据、loss、训练预算，只替换 FlowAssembly | iMonotone `Final=82.33`，AffineCoupling `80.03`，差约 `+2.30`；只是整个可逆层族的消融，参数量未严格对齐 | `SCORE` + `DIAG` |
| `A-ARCH-02` | 训练 loss 是主要差距来源 | 同架构比较 EMD、纯 Chamfer、Chamfer+L2 | 三种 loss 最大差约 `0.46` Final，不能支持“EMD 单独决定高分” | `NEGATIVE` |
| `A-ARCH-03` | Jittor iMonotone 前向无法复现 | source-exact forward、RoundTrip、RootFind 和转换权重对拍 | 张量级误差约 `5e-6~1e-5`；转换权重 mock20 约 `84.425`，与 PyTorch 约 `84.43` 对齐 | `GATE` + `SCORE`，但 `research-only` |
| `A-ARCH-04` | Jittor Light 恢复单调块是否值得更换基础结构 | iMonotone 使用 coeff=0.9、ELU、暖启动后 5 步 Banach 近似梯度；全量清单、seed=123；结构×损失的 20 轮组合对照及 100 轮长训 | mock20：20 轮 Chamfer+0.1L2 下 iMonotone/仿射耦合为 `82.04/81.82`，纯 Chamfer 下为 `81.26/80.66`；100 轮 Chamfer+0.1L2 下均为 `83.26`，仿射耦合为历史模型重评 | `SCORE`；保留历史分数；名义单调身份与参数清单冲突，暂停结构因果解释，见结构身份核查 |
| `A-ARCH-05` | 更贴近原版的 EMD + iMonotone 从零训练可以复现高分 | Jittor 从零训练；原生 EMD、coeff=0.98、LipSwish、5 步 Banach 近似梯度；先小样本过拟合，再全量训练 | 工程与恢复检查通过；续训 ep3–19 无崩溃，mean EMD 在 `1.06~1.07` 震荡；ep19 `mock20 Final=78.81`，低于 `82.5` 停止线 | `NEGATIVE`；本次训练配方未达分数要求，非算子未实现 |
| `A-ARCH-06` | 上一条失败只是 epoch 不够 | 该实验未追加调参；复核 overfit-5、保守配方长训和 RootFind 残差 | overfit-5 能降 3.5+ 个数量级；保守配方可训；高 coeff 的 RootFind 收敛又不充分 | `DIAG` |
| `A-ARCH-07` | MLGC 的感受野、全局模块或宽度是主要瓶颈 | `K=64`、Hilbert global block、hidden `64->128` 的 matched probe | `width128` mock20 `75.29` vs control `74.18`，但 coverage `0.5706` vs `0.5782`；Hilbert `74.35`、coverage `0.5765`；K64 diagnostic coverage `0.5878`，仍未接近成熟基座 | `WEAK_POSITIVE` |
| `A-ARCH-08` | 更强的 GroupToken/全局混合 backbone 可以保住早期 coverage 优势 | GroupToken-HilbertAttention、PointIdentity bypass、再叠加 FCD 的 1k 长训 | GroupToken 早期 coverage `0.5902` 但 Final `34.69`；加入 bypass 后长训最高约 `62.30`，FCD scratch 最高 `68.44` 且 coverage 降到 `0.5708` | `NEGATIVE` |
| `A-ARCH-09` | 显式 direction head 能突破隐式 displacement 平台 | replace-output、dir-only、stable-mag、context-tap、auxiliary 多个 matched probe | replace-output 直接 identity collapse；dir-only 的 `disp_cos` 升至约 `0.37~0.38`，但最佳 Final 约 `68.71`；auxiliary 最好约 `77.54`，仍低于 matched control `78.07` | `NEGATIVE` + `DIAG` |
| `A-ARCH-10` | 原版参考模型或 Flow-Denoise-Lite 可以直接替代当前基座 | 受控 reference Light/Heavy direct transfer；Flow-Denoise-Lite 多数据、多 epoch scaling | reference Light/Heavy 只有 `72.73/71.65`；Flow-Denoise-Lite cloud scaling 最好 `mock200 Final=81.77`，仍低于 `83.72` 对照；clean-oracle 可到 `90.88` 但不可部署 | `NEGATIVE` + `ORACLE` |
| `A-ARCH-11` | Heavy+auction EMD 的 patch loss 下降最终会转成 full-cloud 分数 | reference Heavy、auction EMD、不同训练样本量和 step 数的 official mock20 | patch loss ratio 可降到约 `1e-5`，但 Final 只有 `72.07/71.88/71.01`，低于 Heavy control `72.82`；更多数据和 step 也没救回分数 | `NEGATIVE` |

### 12.1 EMD/iMonotone 失败的准确归因

从零训练实验使用的配置为：

> `coeff=0.98 + 可学习 LipSwish + EMD-only + 5-step Banach surrogate` 这一具体配方，在 full15k、17 个有效训练 epoch的 Jittor 实例中没有兑现小规模 overfit 或保守配方的收敛水位。

该组合未通过预设的训练与评分阈值，停止使用。另有前向对齐、小规模 EMD 训练和保守 iMonotone 配方长训通过的记录，因而本次负结果限于上述组合。

## 13. `A-CASCADE`：二遍推理、融合与专训

本组比较同模型重复推理、第一遍与第二遍结果融合，以及针对训练集第一遍输出训练 specialist 的结果。

| ID | 假设 | 方法 | 结果 | 结果类型 |
|---|---|---|---|---|
| `A-CASCADE-01` | 第一遍输出已经比 noisy 更适合再次去噪 | 用同一个全覆盖 Light 基座处理第一遍输出，形成 parent-child iter2 | raw `84.06` -> iter2 `84.77`，P2S 明显提高 | `SCORE` |
| `A-CASCADE-02` | 二遍可以适当缩小步长，缓解过度精修 | 同点位 parent-child blend，预设权重范围 | `mock200=84.91 / CD=74.80 / P2S=95.01`；online `82.30` | `SCORE` |
| `A-CASCADE-03` | 自迭代可以无限继续 | iter3 继续处理 iter2 输出 | `mock200=83.85 / CD=73.06 / P2S=94.63`，低于 iter2 | `NEGATIVE` |
| `A-CASCADE-04a` | specialist 从少量 shape 开始就能获得真实线上收益 | 约 80 shape 的第一遍输出缓存；specialist raw 与 parent-child blend 对照 | mock200 raw `85.56 / CD=75.89 / P2S=95.23`；blend `85.62 / 75.90 / 95.33`；online blend `83.12 / 72.77 / 93.47` | `SCORE` |
| `A-CASCADE-04b` | 扩大 specialist 数据规模仍能稳定提高 | 约 400 shape；保持第一遍缓存和推理契约，比较 raw/blend | mock200 raw `86.05 / 76.52 / 95.57`；blend `86.08 / 76.50 / 95.66`；online blend `83.55 / 73.36 / 93.73` | `SCORE` |
| `A-CASCADE-04c` | 再扩大到约 1600 shape 仍有可用边际收益 | 约 1600 shape；同时升级 batch，online 使用 raw | mock200 raw `86.39`；online raw `83.81 / CD=73.75 / P2S=93.88` | `SCORE` |
| `A-CASCADE-04d` | 全量第二遍训练能继续兑现小幅收益 | 约 2000 shape；同时改变 optimizer 初始化、epoch 和学习率，raw 与 EMA 对照 | mock200 raw `86.45 / 77.00 / 95.90`；online raw `83.86 / 73.81 / 93.90`；EMA 不推进 | `SCORE` |
| `A-CASCADE-04e` | specialist 的收益主要来自修复 clean-to-pred 覆盖尾部 | 对 specialist 与 iter2 做 residual direction、coverage tail 和全云分解 | CD 改善主要落在 c2p coverage 方向；已测 pred 特征对目标移动的可观测性很弱 | `DIAG` |
| `A-CASCADE-05` | 给二遍 c2p 尾部额外加权可以继续提高 CD | tail-aware c2p loss，与 specialist raw matched 对照 | `Final=86.33` vs 对照 `86.45`；CD、P2S 安全线同时被破坏 | `NEGATIVE` |
| `A-CASCADE-06` | 第二遍容量继续翻倍就能继续提升 | coupling hidden 64 -> 128 及低显存修正版 | first-pass 下降，修正版更低；关闭当前实例 | `NEGATIVE` |

### 13.1 采用专训级联

第二遍专训使用训练集第一遍输出作为输入，对应的干净点云作为目标：

```text
第一遍: noisy -> 粗去噪输出
第二遍: 第一遍输出 -> 专训模型 -> 精修输出
融合:   同点位 parent-child 插值，控制第二遍修正幅度
```

零训练二遍的 mock200 为 84.77，三遍降至 83.85；专训配置为 85.56 至 86.45，均高于零训练二遍。最终采用基础模型与专训模型级联。各专训配置包含数据量和训练超参数变化，未将收益单独归因于某一项变化。

## 14. 最终 A 榜配置与检查

提交配置要求及检查项目：

| 筛选条件 | 处理方式 |
|---|---|
| 输入输出点数一致 | 任何会删点、FPS 下采样或改变 shape 的后处理都不能进入提交链 |
| 推理只使用部署期可见数据 | clean、mesh、normal 真值和 oracle assignment 只留在评测/诊断侧 |
| 完整点云评测 | patch loss、paired cosine、coverage proxy 只能解释，不能替代 Final/CD/P2S |
| 同时报告 CD 和 P2S | 单项上涨但另一项破坏安全线的路线关闭 |
| 记录权重与预测来源 | checkpoint、预测、评测、打包和哈希通过 manifest 串联 |
| 复现范围 | 新环境可以复现流程，但不承诺 GPU 浮点和历史随机点云逐位一致 |

最终 A 榜推理流程：

```text
带噪测试点云
  -> 基础模型
  -> 评分归一化损失训练的 specialist
  -> 几何方向场后处理
  -> 完整 shape / finite / coverage / zip 检查
```

A 榜公开快照的最终线上成绩是 `84.10 / CD 74.08 / P2S 94.13`。B 榜公开复现链则在另一套数据和训练契约下独立组织，见 [experiment_journey.md](overview.md) 与 [A/B 榜复现说明](../b_board/reproduction.md)。

## 15. 随包证据与未包含文件

| 内容 | 公开版状态 | 说明 |
|---|---|---|
| A 榜最终代码 | 随 `a_board/` 提供 | 可按 README 准备数据后复现 |
| B 榜最终代码和权重 | 随 `b_board/`、`checkpoints/` 提供 | 可运行最终双遍推理链 |
| A 榜历史实验总账 | 本文提供 | 每条实验有独立 ID、设置和结果 |
| 代表方法横评日志 | 随 `method_comparison_evidence.json` 提供 | 11 个主评测节点的原始日志、分数和哈希；不等于随包提供完整 PyTorch 重训环境 |
| Heavy 迁移与 Light 消融证据 | 随 [jittor_adaptation_evidence.json](evidence/jittor_adaptation_evidence.json) 提供 | 11 份原始指标对象、1 组修正范围后的摘要字段、1 份 EMD 评测输出及 4090 失败摘录；不含历史训练环境和全部检查点 |
| 原始训练输出和所有 checkpoint | 不全部随包发布 | 体积大、含内部运行路径；关键数值按记录类型汇总 |
| 失败实验源代码 | 不作为默认入口 | 最终仓库只保留经过复核、能解释当前方法的可运行组件 |
| 线上成绩 | README 和最终结果清单提供 | 线上成绩不由本地 MOCK 冒充 |

未随仓库提供完整配置、运行命令或检查点的历史实验，目前只能查阅本文及所附指标摘要。
