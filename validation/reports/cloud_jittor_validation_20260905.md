# 云端 Jittor 验证记录（2026-09-05）

本文记录公开工作区的 Jittor GPU 验证和一次完整 B 榜双遍推理，不替代 A/B 榜官方线上成绩。
验证使用从公开 `b_board/` 目录同步出的云端隔离副本。

## 1. 环境与范围

| 项目 | 值 |
|---|---|
| Python | 3.9.25 |
| Jittor | 1.3.10.0 |
| CUDA 编译工具链 | 12.2 |
| GPU | NVIDIA GeForce RTX 4090，24 GB 显存 |
| 运行方式 | 逐模块公开测试 + B 榜 200 样本双遍推理 |
| 代码来源 | 公开工作区的 `b_board/` 同步副本及公开 `checkpoints/` |

测试开始和结束时均确认 GPU 未保留显存占用。

## 2. 结果

下列十个测试模块均以退出码 0 完成：

| 测试模块 | 覆盖的公开契约 | 结果 |
|---|---|---|
| `test_pdlts_light_model_smoke` | Light 模型完整结构、默认规格 forward、状态保存和加载 | 通过 |
| `test_pdlts_light_system` | 训练系统、checkpoint（检查点）、strict resume（严格续训）、优化器状态 | 通过 |
| `test_pdlts_light_predict_system` | 推理分级、输出清单、提交打包和提交闸门 | 通过 |
| `test_coverage_sweep` | FPS+KNN 覆盖率与 `patch_denoise` 一致性、归一化清单缺失、resume | 通过 |
| `test_normalization_wiring_smoke` | GroupNorm（组归一化）接线、状态字典和优化器更新 | 通过 |
| `test_pdlts_inn_invertibility` | coupling（耦合层）与 12 层 flow 的可逆性和梯度 | 通过 |
| `test_dcd_official_jittor` | DCD 的 Jittor/NumPy 数值等价和梯度 | 通过 |
| `test_safe_knn` | 非整除 batch、单点和历史异常形状的 KNN | 通过 |
| `test_pdlts_light_denoise` | patch 拼接、覆盖率记录、writer（写入器）保护和小规模预测 | 通过 |
| `test_pdlts_light_data_smoke` | 数据 process、训练步骤、transform（变换）链和 predict transform | 通过 |

## 2.1 完整 B 榜推理与打包

在同一云端隔离副本中，使用公开清单对应的两份权重和 200 个官方测试输入完成了完整
级联链：

| 阶段 | 运行标识 | 结果 |
|---|---|---|
| 第一遍 base（基础模型） | `20260905_130655_b_final_pass1_base_official_b_predict` | 200/200，`verdict=green` |
| 第二遍 specialist（专训模型） | `20260905_144204_b_final_pass2_specialist_cascade_official_b_predict` | 200/200，`verdict=green` |
| B 榜提交包 | `20260905_153635_submission` | 200 个 `denoised.npy`，`shape_guard_bad=0` |

第二遍首次按 Jittor 默认 16 路算子编译启动时，在尚未读取样本前发生原生编译器段错误；这
不是模型输出或数据完整性错误。随后使用公开的 `run_jittor_task.py`，将
`JITTOR_COMPILER_THREADS=1` 后从第一遍实际输出缓存重启，第二遍完整通过。因此公开
`reproduce_b_final.sh` 已默认采用单路算子编译，并保留该变量供其他环境调节。

提交包的独立压缩包检查结果：

- `zip_test=None`，压缩包可完整读取。
- 压缩包共 200 个条目，全部为 `shapenet/00000000/<sample_id>/denoised.npy`，无重复条目。
- `result.zip` SHA-256：`2f919fcf6da1ad7f312fbdc1a7bd8a60083228a7d94f9ee973c9f6a8bf361cb4`。
- 200 个输入文件相对路径清单 SHA-256：`b3108823746a35c0c0178a47a0dff5a8b70409b1d70cc5fa441246158da7232a`。

本次使用的公开权重哈希仍与 `b_board/checkpoints/checkpoint_manifest.json` 一致：

- `base_ep149.pkl`：`d9204ea285bf1307fa81d6d63698a4c2b2b3e4b02b934aa689ff26dce791cd8c`
- `specialist_final.pkl`：`9cf3da467828fb8b6cf613ee69abd5ad50a51e09430a83df22174a8265e21bf9`

可追溯的关键观测：

- 归一化测试确认候选 GroupNorm 路径有 49 个归一化层，运行状态条目为 0；作为对照的
  BatchNorm 路径有 98 个运行状态条目。两条路径均可完成有限值 forward 和优化器更新。
- 12 层 flow 的正反向残差均在约 `1e-6` 量级。
- DCD 的 Jittor 与 NumPy 实现的最大示例差值为约 `3e-8`。
- 打包测试验证：green/yellow 才可默认打包；形状不符、red/unknown verdict（结论）和
  样本不全会被拒绝；输出固定到 `outputs/submissions/b_final/<submit_id>/`。

## 3. 验证中修正的问题

| 发现 | 风险 | 公开工作区修正 |
|---|---|---|
| 覆盖率扫描脚本没有进入公开 `scripts/shared/`，且原文件存在未定义的项目根函数 | 覆盖检查无法启动，且无法审计 patch 覆盖 | 迁入并修正为从脚本位置寻找公开代码根；强制显式 `--stage`，按 `outputs/evals/<stage>/<eval_id>/` 归档 |
| 三个测试仍引用搬迁前的脚本或配置路径 | 代码已经迁移但验证会误报失败 | 测试统一引用 `scripts/shared/` 和 `configs/transform/_shared/` 的公开路径 |
| B 榜打包器文档声明阶段目录，非标准输入却会从父目录猜测阶段 | 临时目录或错误命名可能使实际产物违反公开目录契约 | 打包器固定写入 `outputs/submissions/b_final/<submit_id>/`，并在清单写入 `stage=b_final` |

## 4. 未覆盖范围

此次验证未向官方平台提交，也没有用测试集 clean/oracle（干净点云或答案）计算分数；因此
不能从这次运行推导新的线上成绩。公开包不包含赛题训练数据，`test_pdlts_light_data_smoke`
中依赖 `dataset/train/` 的真实 OBJ 分支明确显示为 skip（跳过）；从零训练也未在该次云端
验证中执行。B 榜 200 样本推理、输出完整性、数值有效性和结果压缩包已完成验证。

后续已补充 A/B 榜合成黄金样例与完整性检查，见 [发布准备与验证](../../docs/release_preparation.md)。
本记录中的 B 榜实际双遍推理和结果包验证已覆盖公开 checkpoint 复现链，但不等同于官方线上重新提交。

本页是当时云端副本的验证记录。后续发布包因上游未明确授权而排除了 DCD 对照实现及测试，
此处 DCD 数值结果保留为历史观察；不表示当前发布包仍提供该模块。
