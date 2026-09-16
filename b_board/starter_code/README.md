# 模型、配置与测试

本目录包含 B 榜可执行代码、配置、datalist（样本清单）和测试。
训练编排与评测脚本位于 `../scripts/`，模型和数据处理组件位于 `src/`。

## 运行任务

从本目录执行：

```bash
python run.py --task configs/task/b_final/train_base_scratch_ep100.yaml
```

配置按 `configs/<kind>/<stage>/` 组织，共用配置放在 `configs/<kind>/_shared/`。
`configs/task/b_final/` 提供 B 榜基础模型训练任务，
推理系统配置位于 `configs/system/b_final/`，完整推理入口为
`../reproduce_b_final.sh`。

A 榜代码与配置位于 `../../a_board/`。
`tests/` 包含配置、数据处理、模型与输出验证，不替代完整重训。

## 文档

- [复现指南](../reproduction.md)
- [实验结果与方案选择](../../experiments/overview.md)
- [代码来源与发布范围](../../docs/release_scope.md)
