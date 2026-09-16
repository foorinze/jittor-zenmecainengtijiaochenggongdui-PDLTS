# 目录与配置名称

## 榜单与模型名称

| 名称 | 含义 |
|---|---|
| `a_final` | A 榜最终方案及对应输出目录 |
| `b_final` | B 榜最终训练和推理配置 |
| `base_ep149` | 完成 epoch 149 的第一遍基础模型 |
| `deep_specialist` | 训练 50 轮的第二遍专训模型 |
| `short_specialist` | 较短训练预算的第二遍对照 |
| `coupling128_capacity_probe` | 耦合层隐藏维度由 64 增至 128 的容量对照 |
| `scratch_dynamic_augmentation_probe` | 从零训练时动态增强的对照 |
| `groupnorm_probe/full_gate` | 分组归一化的小规模和完整训练对照 |

后三类名称用于实验记录，不是最终复现入口。

## 主要路径

| 职责 | 路径 |
|---|---|
| B 榜随包权重推理 | `b_board/reproduce_b_final.sh` |
| B 榜完整训练 | `b_board/scripts/experiments/b_final/train_b_final_pipeline.sh` |
| B 榜训练点云对生成 | `b_board/scripts/experiments/b_final/generate_b_training_pairs.py` |
| B 榜第一遍输出缓存 | `b_board/scripts/experiments/b_final/generate_firstpass_cache.py` |
| B 榜第二遍专训 | `b_board/scripts/experiments/b_final/train_deep_specialist.py` |
| B 榜网格采样工具 | `b_board/scripts/experiments/b_final/mesh_sampling.py` |
| B 榜训练任务配置 | `b_board/starter_code/configs/task/b_final/` |
| B 榜第二遍训练样本清单 | `b_board/starter_code/datalist/b_final_specialist_train2k.txt` |
| A 榜独立实现 | `a_board/` |
| A 榜生成的训练点云对 | `a_board/dataset/a_final_train_full15k/` |

配置按 `configs/<kind>/<stage>/` 组织，共用配置放在 `_shared/`。
运行产物按 `outputs/<kind>/<stage>/<artifact_id>/` 保存，
其中 stage（榜单阶段）为 `a_final` 或 `b_final`。
