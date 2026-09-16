# B 榜最终复现实验入口

本目录包含 B 榜数据准备、基础训练和第二遍专训脚本。
运行产物默认进入 `b_board/outputs/runs/b_final/` 与 `b_board/outputs/predictions/b_final/`。

## 入口

- `train_b_final_pipeline.sh`：从官方 B 榜训练数据重训完整两阶段模型。
- `train_base_pipeline.sh`：只训练第一遍 base 模型。
- `generate_b_training_pairs.py`：从官方 B 榜训练 mesh 生成稳定种子的 noisy/clean 点云对。
- `generate_firstpass_cache.py`：生成第二遍 specialist 所需的第一遍输出缓存。
- `train_deep_specialist.py`：训练最终第二遍 specialist 模型。
