# 通用运行工具

本目录提供 A/B 本地评测、结果打包、输出目录管理、覆盖率检查和 Jittor 任务启动工具。

`run_mock_eval.py` 是 A/B 榜共用的本地 MOCK 驱动。它通过 `--stage` 和 `--starter-root`
选择 A 榜或 B 榜评测器；传入 `--scope` 后检查对应的 20/200 样本数。

`coverage_sweep.py` 是 A/B 榜共用的几何覆盖检查。它不执行网络 forward，而是按最终
`patch_denoise` 的 FPS+KNN 规则计算 coverage；必须显式传入 `--stage`，产物写入
`outputs/evals/<stage>/<eval_id>/`。

`run_jittor_task.py` 启动 B 榜配置任务。它默认把
`JITTOR_COMPILER_THREADS` 设为 1，规避部分 Jittor/CUDA 环境连续启动大模型时的并行编译
崩溃；该设置只作用于算子编译阶段。

`pack_submission.py` 检查预测点数、有限值、样本完整性和预测清单，
通过后生成比赛提交用的 `result.zip`。
`output_layout.py` 管理 `outputs/<kind>/<stage>/<artifact_id>/` 下的运行路径。
