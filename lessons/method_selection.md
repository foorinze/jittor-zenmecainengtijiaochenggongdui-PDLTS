# 方法横评与 Light/Heavy 选择

## 2. 方法横评与路线选择

### 2.1 横评结果先说明配置差异

ScoreDenoise、StraightPCF、PD-LTS Light 和 Heavy 都需要适配本任务的输入点数、噪声构造、训练数据和完整点云评测。20 轮 PyTorch mock20 结果分别为 43.97、70.20、82.68 和 57.89。StraightPCF 使用预训练 CVM 初始化，其余主要为随机初始化；训练预算、加噪方式和硬件并不完全相同。

**经验**：这组实验支持“在已测配方中选择 PD-LTS Light”，不能写成等算力、充分收敛的方法能力排名。ScoreDenoise 的训练损失下降而后期分数回落，说明推理步长和完整评分需要分开检查；StraightPCF 的短训节点高于后续节点，也不能据此判定方法上限。

### 2.2 Light/Heavy 是分数、成本和可执行性的联合决策

- Light 20 轮 mock20 为 82.68，记录耗时约 5.9 小时。
- Heavy 20 轮 mock20 为 57.89，记录耗时约 17.8 小时，使用三阶段监督。
- Jittor Heavy 经历 CUDA 非法地址访问，以及 RTX 4090 在 `root_find -> find_fixed_point` 路径的 `CUBLAS_STATUS_EXECUTION_FAILED`；短周期 micro49/59/79 得分为 70.28/72.35/72.82，但计划内长训没有完成。
- 补充 Heavy + EMD 小规模运行未超过继续扩量门槛。

**经验**：减小 batch 没有自动解决 Heavy 长训问题；报错栈只能定位失败路径，不能把根因写成显存不足或某个固定缺陷。现有结果没有排除 Heavy 的后程优势，最终选 Light 是在已测分数、训练成本和迁移稳定性下的工程决策。

**证据**：[实验总览第 3 节](../experiments/overview.md)、[Jittor 迁移与消融证据](../experiments/evidence/jittor_adaptation_evidence.json)。



[返回经验目录](README.md) · [查看分类实验](../experiments/README.md)
