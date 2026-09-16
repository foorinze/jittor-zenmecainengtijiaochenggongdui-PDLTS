# A 榜到 B 榜的算法改动

## 1. 成绩与代码入口

| 项目 | A 榜 | B 榜 |
|---|---|---|
| 名次 | 第 5 名 | 第 8 名 |
| 总分 | 84.10 | 81.05 |
| CD | 74.08 | 70.01 |
| P2S | 94.13 | 92.10 |
| 复现入口 | `a_board/run_all.sh` | `b_board/reproduce_b_final.sh` |

A 榜原代码和权重完整保留在 `a_board/`。B 榜没有覆盖 A 榜快照。

## 2. 保持不变的部分

- 任务仍是输入带噪三维点云，输出点数不变的去噪点云。
- 主体网络仍是 PDLTSLight，使用局部图特征、12 层可逆流和潜空间噪声通道切除。
- 第一遍和第二遍使用同一网络结构，第二遍从第一遍权重 warm-start（暖启动）。
- 推理只读取 noisy 输入和模型权重，不使用测试集 clean、mesh、标签或由它们生成的缓存。
- 训练、推理和打包均使用 Jittor 实现。

## 3. 数据改动

### A 榜

A 榜使用原 A 榜训练数据。最终方案包含 base、score-normalized specialist（评分归一化损失的第二遍专训模型）和几何方向场后处理。

### B 榜

B 榜改用 B 榜官方 train split：

- 原始 archive 共 19,799 个 shape。
- 官方 train split 为 19,699 条，validate split 为 100 条。
- mesh 健康检查后，实际训练使用 19,528 个样本。
- B 榜类别在数据配置中压平为 `00000000`。
- 每个健康 mesh 生成 50,000 点训练对，噪声为 sigma 0.0075 到 0.0161 的 Laplace 噪声。
- 训练输入使用预生成的 `clean.npy/noisy.npy`，训练 transform 不重复做 mesh 采样、归一化或加噪。

这些操作只使用官方训练数据，与 B 榜测试集无关。

## 4. 第一遍 base 改动

| 项目 | A 榜 | B 榜 |
|---|---|---|
| 起点 | A 榜训练链 | 在 B 榜训练数据上从零开始 |
| 主训练 | A 榜原配方 | 100 epochs，Adam，lr 5e-4，batch 8 |
| 延长训练 | A 榜原配方 | strict resume 50 epochs，得到 ep149 |
| loss | A 榜原配方 | Chamfer 1.0 + L2 0.1 |

B 榜 ep100 到 ep149 使用 `resume_state` 恢复完整训练状态，不是仅加载模型参数。

## 5. 第二遍 specialist 改动

A 榜最终第二遍包含 score-normalized loss，并在后续使用几何方向场做一步无网络后处理。B 榜最终方案没有沿用这两项：

- B 榜 second-pass specialist 从 B 榜 ep149 base warm-start。
- 使用固定 2,000 个训练 shape。
- 训练 50 epochs，Adam，lr 1e-4，batch 16。
- 每个 shape 每个 epoch 重新采样 20 个 1,024 点 patch。
- patch 以 FPS seed 点中心化。
- loss 改回普通双向 Chamfer L2，不使用 A 榜的 score-normalized loss。
- B 榜最终提交不应用 A 榜几何方向场后处理。

因此，B 榜 specialist 不是 A 榜 specialist 的直接续训，也不是把 A 榜 checkpoint 换到新测试集上推理，而是基于 B 榜 base 和 B 榜训练数据重新训练。

## 6. 推理改动

B 榜最终只保留两遍网络级联：

```text
B official noisy
  -> B base ep149
  -> B second-pass specialist
  -> pack_submission.py
```

两遍 official 推理均固定：

```yaml
seed_k_l1: 12
seed_k_l2: 24
```

specialist 训练缓存使用 12/16 和 `predict_seed_k_alpha=80`。这是生成训练输入时的历史执行口径；最终 official 推理改为 12/24，以保证 200 个样本全部通过完整性检查。`predict_seed_k_alpha` 只影响显存分批。

B 榜最终级联不启用 postselect、候选筛选、refine、pull head 或几何方向场。

## 7. Checkpoint 与结果对应关系

| 阶段 | 包内文件 | 作用 |
|---|---|---|
| 第一遍 | `b_board/checkpoints/base_ep149.pkl` | B 榜 ep149 base |
| 第二遍 | `b_board/checkpoints/specialist_final.pkl` | 从 ep149 warm-start 的 B 榜 second-pass specialist |

ep99 只对应早期 79.41 的单遍结果，不是 81.05 的最终权重。正式复现入口不会读取 ep99 配置。

最终线上结果来自 ep149 第一遍输出再经 second-pass specialist 的第二遍级联，得分 81.05（CD 70.01 / P2S 92.10）。

## 8. 可复现性说明

随包 checkpoint 是最终线上提交使用的实际模型，可用于复现 B 榜最优推理结果。

原始 B 榜训练对在 2026-08-11 生成时使用 Python `hash()` 且没有固定 `PYTHONHASHSEED`，所以旧训练对无法从 mesh 逐位重建。当前重训脚本已改用稳定 SHA-256 种子；它提供完整、可重复执行的重训流程，但新权重不保证与随包 checkpoint 逐位一致。
