# B 榜最终方案与复现

B 榜线上总分 81.05（CD 70.01 / P2S 92.10），第 8 名。采用 ep149 基础模型和 2000 个训练形状、50 轮的第二遍专训模型，不使用 A 榜评分归一化损失与方向场后处理。

| 路径 | 内容 |
|---|---|
| `checkpoints/` | 两份最终权重与 SHA-256（哈希校验值）清单 |
| `starter_code/` | 模型、损失、训练推理流程、配置、数据清单与评测器 |
| `scripts/` | 数据准备、基础训练、第一遍缓存、第二遍专训和结果校验 |
| `results/` | 历史最终训练、预测与提交的来源清单 |
| `reproduce_b_final.sh` | 随包权重双遍推理入口 |

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
bash b_board/reproduce_b_final.sh --help
```

测试数据放入 `b_board/dataset/test_noisy_b/`。完整的权重推理和从零训练命令见 [复现指南](reproduction.md)，历史权重与结果的对应关系见 [结果清单](results/README.md)。

[A/B 差异](../docs/a_to_b_changes.md) · [实验记录](../experiments/b_board_records.md) · [经验总结](../lessons/README.md)
