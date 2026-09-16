# B 榜 81.05 复现指南

本文分别说明如何使用随包 checkpoint（检查点）复现 B 榜提交，以及如何从官方原始训练数据重新训练。提交包不含数据集。

A/B 榜 MOCK、黄金样例、归一化和失败实验的统一口径见
`docs/evaluation_protocol.md`。本指南只补充 B 榜最终链路和
重训边界，不把 MOCK 分数写成线上成绩。

## 1. 结果与入口

| 项目 | 内容 |
|---|---|
| 赛道 | 赛道二 |
| 队伍 | 怎么才能提交成功队 |
| B 榜名次 | 第 8 名 |
| B 榜得分 | 81.05（CD 70.01 / P2S 92.10） |
| checkpoint 推理入口 | `b_board/reproduce_b_final.sh` |
| 完整重训入口 | `b_board/scripts/experiments/b_final/train_b_final_pipeline.sh` |
| A 榜原复现入口 | `a_board/run_all.sh` |

最终 B 榜模型链：

```text
official B noisy
  -> B base ep149
  -> second-pass specialist
  -> result.zip
```

两遍 official 推理都使用 `seed_k_l1=12`、`seed_k_l2=24`。

## 2. 环境

目标环境与比赛要求一致：

- Ubuntu 22.04
- NVIDIA RTX 4090
- CUDA 12.x（云端完整验证使用 12.2）
- Python 3.9.25（已测环境，与 `environment.yaml` 一致）
- Jittor 1.3.10

创建环境时，在包根目录选择一种方式：

```bash
python -m pip install -r requirements.txt
```

或：

```bash
conda env create -f environment.yaml
conda activate pcdenoise
```

Jittor 首次导入会编译 CUDA 算子，需要可用的 `g++` 和 CUDA toolkit。代码未使用 JittorGeometric、Pandas 或 scikit-learn，因此没有添加这些无效依赖。

## 3. 数据准备

### 3.1 B 榜测试集

将官方 B 榜测试数据放在：

```text
b_board/dataset/test_noisy_b/
  shapenet/
    00000000/
      <sample_id>/
        noisy.npy
```

测试集共 200 个样本。推理只读取 `noisy.npy` 和模型权重，不读取 `clean.npy`、mesh、norm、oracle 或测试标签。

### 3.2 B 榜训练集

从头训练时，将官方 B 榜原始 mesh 放在：

```text
b_board/dataset/train_b/
  shapenet/
    00000000/
      <sample_id>/
        models/model_normalized.obj
```

数据数量口径如下：

| 口径 | 数量 |
|---|---:|
| B 榜原始 archive | 19,799 |
| 官方 train split | 19,699 |
| 官方 validate split | 100 |
| 项目健康检查通过并用于训练 | 19,528 |

`b_board/starter_code/datalist/train_b.txt` 有 19,699 条，对应官方 train split。训练数据生成脚本逐个检查 mesh，只把 19,528 个健康且生成成功的样本写入 `train_b_generated.txt`；数量不符会以非零状态退出。

## 4. 使用随包权重复现 B 榜提交

包内权重：

| 文件 | SHA-256 |
|---|---|
| `b_board/checkpoints/base_ep149.pkl` | `d9204ea285bf1307fa81d6d63698a4c2b2b3e4b02b934aa689ff26dce791cd8c` |
| `b_board/checkpoints/specialist_final.pkl` | `9cf3da467828fb8b6cf613ee69abd5ad50a51e09430a83df22174a8265e21bf9` |

从包根目录执行：

```bash
bash b_board/reproduce_b_final.sh
```

公开入口默认使用单路 Jittor 算子编译，以规避部分环境在连续启动两遍大模型时的原生
编译崩溃。需要调整时可显式设置，例如：

```bash
JITTOR_COMPILER_THREADS=1 bash b_board/reproduce_b_final.sh
```

该变量只影响首次算子编译并发，不改变模型结构、参数或推理逻辑。

脚本执行以下步骤：

1. 检查两个 checkpoint 和 200 个测试样本。
2. 使用 ep149 base 完成第一遍推理。
3. 检查第一遍 200 个 `denoised.npy` 和 `manifest.json` 的 `verdict=green`。
4. 将第一遍实际输出目录写入临时 cascade（级联）配置。
5. 使用 second-pass specialist 完成第二遍推理。
6. 再次检查 200 个输出和 `verdict=green`。
7. 调用 `scripts/shared/pack_submission.py` 生成 `result.zip`。

临时 task/data 配置在脚本退出时自动删除，正式配置不会被改写。输出位于 `b_board/outputs/` 下，由底层配置自动生成预测目录和提交目录：

```text
b_board/outputs/predictions/b_final/<run_id>/
b_board/outputs/submissions/b_final/<submit_id>/result.zip
```

最终打包检查应满足：

```text
n_samples=200
n_missing_total=0
shape_guard_bad=0
verdict_at_pack=green
force=false
```

## 5. 从原始训练数据重训

### 5.1 一键入口

数据就位后执行：

```bash
cd b_board
bash scripts/experiments/b_final/train_b_final_pipeline.sh
```

这是公开仓库的完整重训入口。它会自动生成
`starter_code/datalist/train_b_generated.txt`，无需手工创建清单。
历史云端执行脚本仅作为过程证据保留，不作为公开复现入口。

流水线依次完成以下四步。

### 5.2 生成训练点云对

入口：

```bash
python scripts/experiments/b_final/generate_b_training_pairs.py \
  --dataset-dir dataset/train_b \
  --datalist starter_code/datalist/train_b.txt \
  --output-datalist starter_code/datalist/train_b_generated.txt \
  --expected-success 19528 \
  --num-workers 8
```

每个健康 mesh 生成 50,000 点 `clean.npy` 和 Laplace 加噪的 `noisy.npy`，并保存 `norm.json`。噪声 sigma 范围为 0.0075 到 0.0161。默认以样本相对路径的 SHA-256 生成稳定种子。

### 5.3 训练第一遍 base

入口：

```bash
bash scripts/experiments/b_final/train_base_pipeline.sh
```

关键参数：

| 阶段 | epochs | batch size | optimizer | learning rate | loss |
|---|---:|---:|---|---:|---|
| scratch | 100 | 8 | Adam | 5e-4 | Chamfer 1.0 + L2 0.1 |
| strict resume | 50 | 8 | Adam | 5e-4 | Chamfer 1.0 + L2 0.1 |

每个训练 patch 为 1,024 点。第二段通过 ep99 的 `.train.pkl` 恢复模型、优化器和随机状态，运行 ep100 到 ep149；这一步是 strict resume，不是仅加载模型权重的 warm-start。

### 5.4 生成 specialist 训练缓存

`scripts/experiments/b_final/generate_firstpass_cache.py` 使用刚训练的 ep149 base，对固定 2,000 个 shape 生成第一遍输出。为控制显存和避免长进程碎片化，脚本按 150 个样本分块推理后合并。

训练缓存沿用实际训练口径：

```text
seed_k_l1=12
seed_k_l2=16
predict_seed_k_alpha=80
```

`predict_seed_k_alpha` 只控制 patch 的显存分批，不改变模型参数和 patch 集合。训练缓存的 12/16 与最终 official 推理的 12/24 用途不同。

### 5.5 训练 second-pass specialist

specialist 从 ep149 base warm-start，关键参数：

| 参数 | 值 |
|---|---:|
| 训练 shape | 2,000 |
| epochs | 50 |
| batch size | 16 |
| learning rate | 1e-4 |
| 每 shape 每 epoch patch 数 | 20 |
| patch size | 1,024 |
| seed | 42 |
| EMA | 关闭 |
| loss | 普通双向 Chamfer L2 |

训练每个 epoch 重新采样 patch，以 FPS seed 点作为 patch 中心。输出 checkpoint 为：

```text
b_board/outputs/runs/b_final/<specialist_run>/b_specialist_final.pkl
```

### 5.6 使用新训练权重推理

流水线末尾会打印 `BASE_CKPT` 和 `SPECIALIST_CKPT`。回到包根目录执行：

```bash
bash b_board/reproduce_b_final.sh \
  --base-ckpt b_board/outputs/runs/b_final/<base_run>/checkpoints/pdlts_light_149.pkl \
  --specialist-ckpt b_board/outputs/runs/b_final/<specialist_run>/b_specialist_final.pkl
```

## 6. A 榜复现快照

`a_board/` 是 A 榜独立复现快照，保留自己的代码、配置、权重和 `run_all.sh`。A 榜复现说明以本仓库 Markdown 文档为准。

A 榜与 B 榜是两条独立复现入口，不要混用 checkpoint 或配置。具体改动见 `A_TO_B_CHANGES.md`。

A 榜本地 MOCK 需要使用 A 榜快照内的 `evaluate_mock.py`、A 榜 mesh 和每个样本的
`norm.json`；不要直接调用 B 榜 evaluator。完整命令见
`docs/evaluation_protocol.md` 的“2.1 A 榜 MOCK”。

## 7. 可复现性边界

随包两个 checkpoint 是线上 81.05 使用的实际权重，使用第 4 节可复现对应推理链。

2026-08-11 首次生成 B 榜训练对时，旧脚本用 Python `hash()` 派生样本种子，运行环境没有设置 `PYTHONHASHSEED`。因此仅凭原始 mesh 无法逐位重建当时生成的 `clean.npy/noisy.npy`，也不能承诺重新训练得到与随包权重逐位相同的参数。

本包把重训入口改为稳定的 SHA-256 派生种子，保证训练对生成规则跨进程稳定。即使训练对固定，GPU 算子和浮点累加仍可能使重新训练的权重与线上 checkpoint 有细微差异。代码检查若目标是核对线上最优结果，应优先使用第 4 节的随包 checkpoint；第 5 节用于核对从原始训练数据到新权重的完整训练链。

## 8. 最终运行证据

`results/` 保存线上最优链对应的清单：

- `firstpass_prediction_manifest.json`：第一遍 ep149 预测。
- `canonical_cascade_prediction_manifest.json`：第二遍 second-pass cascade 预测。
- `canonical_submission_manifest.json`：最终结果打包。

两份 prediction manifest 的 leakage flags（泄漏标志）均为 `false`。公开版清单已去除历史绝对路径；原始 manifest（清单）保留在最终提交包和内部迁移审计中。
