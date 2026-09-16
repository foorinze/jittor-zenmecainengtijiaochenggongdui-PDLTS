# A 榜最终方案与复现

本目录是 A 榜最终方案的可独立复现快照。详细方法说明、A/B 榜差异和结果边界见
[方法说明](../docs/method_overview.md)、[A/B 差异](../docs/a_to_b_changes.md)及 [实验记录](../experiments/README.md)，这里只讲目录结构与怎么跑。

除本地评测一节另有说明外，以下命令均在 A 榜快照目录执行。从公开仓库根目录进入：

```bash
cd a_board
```

## 1. 放数据

数据集不在代码包内，需要自行放到本目录下：

```text
a_board/
  dataset/
    train/                                    官方训练集（网格）
      shapenet/<类别>/<样本>/models/model_normalized.obj
    test_noisy/                               官方测试集（带噪点云）
      shapenet/<类别>/<样本>/noisy.npy
```

如果磁盘上已有数据集，用软链接指过来即可，不必复制：

```bash
ln -s /你的路径/train      dataset/train
ln -s /你的路径/test_noisy dataset/test_noisy
```

`dataset/a_final_train_full15k/` 由 `run_all.sh` 的第一步自动生成，不需要手工准备。
`infer` 模式只需 `dataset/test_noisy/` 和随包权重；`full` 模式还需 `dataset/train/`。

## 2. 装环境

```bash
conda env create -f ../environment.yaml && conda activate pcdenoise
# 或
python -m pip install -r ../requirements.txt
```

## 3. 跑

```bash
# 从零训练两个阶段，拟合第三阶段系数，再推理打包（单块 4090 约 48 小时）
bash run_all.sh full

# 或：用 checkpoints/ 下的现成权重直接推理打包（约 2.2 小时）
bash run_all.sh infer
```

产物：`outputs/submissions/a_final/repro_a_final_submit/result.zip`

## 4. 目录结构

```text
a_board/
  run_all.sh                     全流程复现入口
  README.md                      本文件
  checkpoints/                       两个阶段的权重 + SHA256 校验值
  starter_code/                  框架主体，训练与推理都从这里进
    run.py                       统一入口，按 task 配置跑训练或推理
    evaluate.py                  官方评测脚本
    evaluate_mock.py             本地验证集评测脚本
    src/
      data/                      数据读取、采样、加噪、patch 切分
      model/pdlts_light/         网络本体
        model.py                 整网装配
        layer.py                 局部几何特征提取
        inn.py                   可逆流模块
        denoise.py               整云推理与 patch 拼接
        system.py                损失函数
      system/pdlts_light.py      训练与推理的流程控制、产物落盘
    configs/                     配置，按 data/model/transform/system/task 分层
      task/repro/                复现用的任务配置
    datalist/                    样本清单
  scripts/
    pipeline/                    数据准备、第二阶段训练、第三阶段拟合与后处理
    shared/pack_submission.py    提交打包与校验
  outputs/                       运行时生成，代码包内为空
```

## 5. 各步骤对应的脚本

| 步骤 | 脚本 / 配置 | 说明 |
|---|---|---|
| 1 | `scripts/pipeline/01_prepare_train_pairs.py` | 从网格采样生成固定 noisy/clean 点云对，供第二阶段做监督目标 |
| 2 | `starter_code/configs/task/repro/01_train_base.yaml` | 第一阶段：基础去噪网络从零训练 100 epoch |
| 3 | `starter_code/configs/task/repro/02_predict_base_official.yaml` | 第一阶段在官方测试集上推理 |
| 4 | `scripts/pipeline/03_generate_firstpass_cache.py` | 第一阶段在 2000 个训练样本上推理，产出第二阶段的训练输入 |
| 5 | `scripts/pipeline/04_precompute_score_weights.py` | 预计算评分归一化损失的逐样本权重 |
| 6a | `scripts/pipeline/05_train_specialist.py --stage main` | 第二阶段主训练，1599 样本 50 epoch |
| 6b | `scripts/pipeline/05_train_specialist.py --stage finetune` | 第二阶段全量微调，2000 样本 12 epoch |
| 6c | `scripts/pipeline/06_train_specialist_scorenorm.py` | 第二阶段评分归一化损失微调，12 epoch |
| 7 | `scripts/pipeline/07a_fit_direction_controller.py` | 拟合第三阶段方向场的 10 个系数（岭回归，约 20 分钟） |
| 8 | `starter_code/configs/task/repro/06_predict_specialist_official.yaml` | 第二阶段在官方测试集上推理 |
| 9 | `scripts/pipeline/07_apply_direction_controller.py` | 第三阶段几何方向场后处理，即最终结果（纯 CPU，约 1 分钟） |
| 10 | `scripts/shared/pack_submission.py` | 校验并打包成 result.zip |

第 7 步只在 `full` 模式跑。`infer` 模式直接用
`07_apply_direction_controller.py` 里的冻结系数，即最终提交所用的系数。从零重训会
得到与我们不同的第二阶段权重，对应的方向场系数也会略有不同，所以 `full` 模式会用
第 7 步自己拟合出的那一组（`run_all.sh` 通过 `--model-json` 传进第 9 步）。

想核对冻结系数是怎么来的，可以在已有第二阶段输出的情况下单独比较系数：

```bash
python scripts/pipeline/07a_fit_direction_controller.py \
    --source <第二阶段在 52 个拟合形状上的输出目录> \
    --verify-frozen
```

`scripts/pipeline/02_make_specialist_datalist.py` 用于生成第二阶段的训练子集清单。
生成好的清单已随包提供（`starter_code/datalist/specialist_train{1600,2000}.txt`），
`run_all.sh` 不重跑这一步；如需核对抽样过程，可单独执行：

```bash
python scripts/pipeline/02_make_specialist_datalist.py \
    --target-n 2000 --seed 42 --out /tmp/check_2000.txt
diff <(sort /tmp/check_2000.txt) <(sort starter_code/datalist/specialist_train2000.txt)
```

## 6. 单步执行

每一步的产物目录名固定，中途失败可以从失败那步单独重跑，不必从头开始。
`run_all.sh` 里每条命令都可以直接复制出来单独执行。训练与推理分别走：

```bash
# 训练 / 推理（都从 starter_code/ 下执行）
cd starter_code && python run.py --task configs/task/repro/<配置名>.yaml

# 数据准备与第二阶段训练（从 A 榜快照目录执行）
python scripts/pipeline/<脚本名>.py <参数>
```

## 7. 本地评测

代码包不含本地验证集。A 榜 MOCK 的完整命令、20/200 样本清单、clean/noisy 黄金
样例和归一化注意事项见仓库根目录的
`docs/evaluation_protocol.md`。这里给出直接调用 A 榜评测器的
最小命令，从公开仓库根目录执行：

```bash
cd a_board/starter_code
python evaluate_mock.py \
    --pred_dir ../outputs/predictions/a_final/<推理目录>/pred \
    --gt_dir ../dataset/mock_test \
    --noisy_dir ../dataset/mock_test \
    --mesh_dir ../dataset/train \
    --datalist datalist/mock.txt \
    --workers 8
```

这里的 `evaluate_mock.py` 要求每个样本有 `clean.npy`、`noisy.npy`、`norm.json`，
并且 mesh 位于与样本键一致的 `models/model_normalized.obj` 路径。缺失预测、点数
不一致和缺失 mesh/norm 都不能静默从均值中删掉。

## 8. 提交路径未使用的代码

框架里保留了几个我们对比过、但**未用于最终提交**的模块。它们都是惰性导入
（只在配置显式开启时才 import），提交所用的配置不会触及：

| 模块 | 内容 | 提交是否使用 |
|---|---|---|
| `src/model/pdlts_light/losses/emd_sum.py` | 最优传输（EMD）损失 | 否 |
| `src/model/pdlts_light/losses/uniformcd.py` | 密度比对应搜索损失 | 否 |
| `src/model/pdlts_light/losses/dcd_*.py` | 密度感知倒角距离损失 | 否 |
| `src/model/emd_jittor/` | 自研 EMD CUDA kernel，仅被 `emd_sum.py` 调用 | 否 |
| `src/model/imonotone_light*/` | 可逆单调块版本的流模块 | 否 |
| `src/model/pdlts_heavy/` | 另一套更大的网络配置 | 否 |

提交实际用到的损失是对称倒角距离（第二阶段）与倒角距离加 L2（第一阶段），
流模块用 ActNorm 加仿射耦合。保留这些模块是为了让 `src/model/parse.py` 的
注册表完整、并保留对比实现的可追溯性。

## 9. 复现注意事项

- **显存**：整云推理峰值约 10GB，训练峰值约 18GB，24GB 卡够用。不要在训练的同时
  跑推理，两者叠加会超出 24GB。
- **首次运行**：Jittor 会即时编译算子，第一次跑会有几分钟编译时间，属正常。
- **随机性**：训练脚本的随机性由 `--seed` 决定（默认 42），patch 抽样与 batch
  打乱用两个独立的随机数发生器。每个 epoch 会把 patch 索引与 batch 顺序的 sha256
  写进 `summary.json`，两次运行可逐位核对。第一阶段训练用 seed 123。
  GPU 浮点累加顺序不保证逐位一致，因此重训分数可能有小数点后二位级别的差异。
- **不使用测试集标签**：全流程只读 `dataset/test_noisy` 下的 `noisy.npy`。
  第二阶段的训练输入是第一阶段在训练集上的输出，监督目标来自训练集网格采样，
  与测试集无关。
- **框架**：训练与推理全程只用 Jittor，不依赖 PyTorch。
