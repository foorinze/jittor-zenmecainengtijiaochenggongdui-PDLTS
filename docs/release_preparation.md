# 发布准备与验证

本仓库以 Git 目录维护。文件清单记录公开源码快照的内容，无需压缩包即可检查；校验状态与托管平台的发布状态分别记录。

## 文件与许可

入口、A/B 复现代码、四份权重、配置、实验记录和验证脚本随包提供。权重逐一核对来源
清单中的 SHA-256（哈希校验值）。原创代码采用 MIT 许可，上游代码及改写保留各自授权；
详见 [第三方来源与许可](../THIRD_PARTY_NOTICES.md) 和 [软件引用](../CITATION.cff)。

DCD 上游没有明确许可证，涉及逐行改写的五个对照文件已列入
[发布排除清单](../validation/release_inventory.py)，同时列入根目录 Git 忽略规则。
这些文件不在当前目录中。模型拒绝启用该实验分支，A/B 最终配置均使用
默认关闭状态。实验指标、失败记录和 86 条 A 榜实验记录保留。

当前目录不包含数据集、预测数组、运行日志、缓存或本地环境。文件清单记录每个文件的
路径、字节数和 SHA-256，自身除外。

## 验证范围

2026-09-16 目录调整后，重新执行 A/B 命令入口、两种后端的评测契约、Jittor CPU 模型及静态检查。依赖解析和云端 GPU 推理仍是既有历史记录，不代表在新目录重新执行了完整训练或 GPU 推理。模型代码的计算逻辑、配置数值、数据清单与四份权重未改变；源文件中的文档路径和运行入口的权重路径已随目录调整。

| 检查 | 结果与证据 |
|---|---|
| Python、Shell、JSON、YAML/CFF 语法及文档链接 | [静态检查](../validation/reports/release_static_validation.json)，包含文件哈希与四份权重校验 |
| 本机路径、内部编号、联系方式、常见密钥格式 | 同上；规则扫描通过不等于任意形式敏感信息均可被自动识别 |
| A/B 评测，NumPy 三角面后端 | [22 个用例](../validation/reports/release_evaluator_validation.json)，Windows Python 3.13.9 |
| A/B 评测，pcu-BVH 后端 | [22 个用例](../validation/reports/release_evaluator_pcu_validation.json)，Linux Python 3.9.25 |
| A/B 命令入口与缺失数据提示 | [7 个用例](../validation/reports/release_entrypoint_validation.json)，未启动训练或推理 |
| 排除 DCD 后的 A/B 模型 | [损失与梯度检查](../validation/reports/release_model_validation.json)，Jittor CPU 合成小点云 |
| 安装依赖解析 | [解析记录](../validation/reports/release_environment_validation.json)，Linux Python 3.9.25 |
| 云端 Jittor GPU 与完整 B 榜推理 | [2026-09-05 验证记录](../validation/reports/cloud_jittor_validation_20260905.md)，含 200 样本双遍推理与压缩包检查 |

合成评测用例包含干净点云预测 100 分、带噪点云预测 0 分，以及缺失预测、点数错误、
非有限预测、缺失 clean/noisy、mesh 或归一化文件。非平凡的网格平移和缩放用于检查
评测器确实应用了归一化参数。这些结果验证评分契约，不是新的模型得分。

指定清单中缺失 clean/noisy 时直接终止评测，避免静默缩小样本集合；
A 榜 `infer`（推理模式）只检查推理数据，`full`（完整模式）还检查训练数据。

## 文件清单与复核

从仓库根目录执行。验证命令和两种评测后端的说明见 [验证脚本说明](../validation/README.md)。

```bash
python -B validation/audit_release.py --output validation/reports/release_static_validation.json
python -B validation/prepare_release.py --version pdlts_source_20260916 --manifest-only
python -B validation/prepare_release.py --version pdlts_source_20260916 --verify-only
```

程序要求所有验证报告通过，并核对验证后的文件集合与哈希未变。
`--manifest-only` 只更新 `manifests/release_file_manifest.json`，不生成压缩包；
`--verify-only` 检查当前文件是否与该清单一致。

GitHub 与 GitLink 使用同一份目录内容，仓库名见根 README。
代码或文档发生改动后，重新执行相关验证并更新文件清单。

## 已知边界

这轮发布验证未创建全新 conda 环境，未执行真实训练 OBJ（网格文件）分支或从零训练；
依赖解析通过不代表完整环境验收。GPU 全量推理来自上述云端记录，后续评测与许可调整
使用针对性验证，没有重跑整套训练。

B 榜历史训练点云对受未固定的 Python `hash()` 影响，无法逐位重建。当前重训脚本使用
稳定种子，但不承诺新权重逐位等同随包权重，详见 [复现指南](../b_board/reproduction.md)。
MOCK（本地模拟评测）结果不能替代官方成绩。
