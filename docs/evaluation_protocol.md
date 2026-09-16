# A/B 榜 MOCK 与归一化规范

本文说明 A/B 榜 MOCK（本地模拟评测）的命令、评分和归一化方式。
MOCK 用于本地比较和回归检查，官方成绩以平台 online（线上）结果为准。

## 1. 评测范围

| 名称 | 用途 | 是否等于线上成绩 |
|---|---|---|
| A 榜 MOCK | A 榜快照的本地回归、清单和输出完整性检查 | 否 |
| B 榜 MOCK / mock_b | B 榜模型选择和失败实验筛选 | 否 |
| A/B 榜 online | 官方平台实际评测 | 是，只有这一项是赛题成绩 |

评测记录包含 `stage`（榜单阶段）、`scope`（评测范围）、样本数和实现。
例如 `B / mock_b / 20 samples / evaluate_mock.py` 表示 20 样本本地评测，
`B / online / 200` 表示 200 样本官方评测，两者的分数不直接比较。

## 2. 统一入口

通用驱动位于 `b_board/scripts/shared/run_mock_eval.py`。它将相对路径解释为 `b_board/`
目录，并把命令、完整日志、指标和清单保存到：

```text
b_board/outputs/evals/<stage>/<eval_id>/
  command.sh
  logs/eval.log
  report.txt
  metrics.json
  manifest.json
```

下面每段命令都从仓库根目录开始，并先进入 `b_board/`。A 榜的参数通过 `../a_board/` 指向同级目录；连续执行时不要重复进入 `b_board/`。

### 2.1 A 榜 MOCK

A 榜数据目录约定为：

```text
a_board/dataset/mock_test/
  shapenet/<类别>/<样本>/noisy.npy
  shapenet/<类别>/<样本>/clean.npy
  shapenet/<类别>/<样本>/norm.json
a_board/dataset/train/
  shapenet/<类别>/<样本>/models/model_normalized.obj
```

若 MOCK 数据尚未有清单，先扫描同时存在 `noisy.npy` 和 `clean.npy` 的样本：

```bash
cd b_board
python scripts/shared/run_mock_eval.py \
  --build-datalist \
  --stage a_final \
  --starter-root ../a_board/starter_code \
  --mock-dir ../a_board/dataset/mock_test \
  --limit 20 \
  --out ../a_board/starter_code/datalist/mock.txt
```

A 榜普通预测的评测命令：

```bash
cd b_board
python scripts/shared/run_mock_eval.py \
  --stage a_final \
  --starter-root ../a_board/starter_code \
  --predict-run ../a_board/outputs/predictions/a_final/<run_id> \
  --mock-dir ../a_board/dataset/mock_test \
  --mesh-dir ../a_board/dataset/train \
  --datalist ../a_board/starter_code/datalist/mock.txt \
  --scope mock20 \
  --eval-id a_final_mock20_<run_id>
```

如果使用全量 A 榜本地清单，将 `mock.txt` 和 `mock20` 分别换成 `mock_full.txt` 和
`mock200`。`mock20` / `mock200` 在驱动层会硬检查总样本数、有效预测数和缺失预测数，
因此不能用部分结果冒充全量 MOCK。

### 2.2 B 榜 MOCK

B 榜历史 MOCK 与 B 榜自建 `mock_b` 的数据位置不同，必须在实验记录中写清楚使用的
数据源。普通 B 榜 MOCK：

```bash
cd b_board
python scripts/shared/run_mock_eval.py \
  --stage b_final \
  --starter-root starter_code \
  --predict-run outputs/predictions/b_final/<run_id> \
  --mock-dir dataset/mock_test \
  --mesh-dir dataset/train \
  --datalist starter_code/datalist/mock_full.txt \
  --scope mock200 \
  --eval-id b_final_mock200_<run_id>
```

B 榜自建验证集使用 B 榜训练数据生成的 `norm.json` 和对应 mesh：

```bash
cd b_board
python scripts/shared/run_mock_eval.py \
  --stage b_final \
  --starter-root starter_code \
  --predict-run outputs/predictions/b_final/<run_id> \
  --mock-dir dataset/mock_test_b \
  --mesh-dir dataset/train_b \
  --datalist starter_code/datalist/mock_b.txt \
  --scope mock_b \
  --eval-id b_final_mock_b_<run_id>
```

`mock_b` 是自建分布，不能把它的分数写成 B 榜 online。它的价值是比较同一批模型、
同一套输出完整性规则和同一套归一化口径。

### 2.3 两个黄金样例

参考样例直接把 MOCK 样本目录当作预测目录，不要求预测运行的 `manifest.json`：

```bash
cd b_board

# clean 作为预测：Final 应接近 100
python scripts/shared/run_mock_eval.py \
  --stage b_final \
  --starter-root starter_code \
  --predict-run dataset/mock_test \
  --mock-dir dataset/mock_test \
  --mesh-dir dataset/train \
  --pred-filename clean.npy \
  --eval-id b_final_golden_clean

# noisy 作为预测：Final 应接近 0
python scripts/shared/run_mock_eval.py \
  --stage b_final \
  --starter-root starter_code \
  --predict-run dataset/mock_test \
  --mock-dir dataset/mock_test \
  --mesh-dir dataset/train \
  --pred-filename noisy.npy \
  --eval-id b_final_golden_noisy
```

A 榜参考样例检查只需把 `--stage` 改为 `a_final`，并把 `--starter-root`、`--mock-dir` 和
`--mesh-dir` 改成 A 榜路径。若 clean-as-pred 不能接近 100，优先检查 mesh、`norm.json`
和点云是否在同一坐标系，不要先调整模型参数。

## 3. 评测实现的准确口径

`b_board/starter_code/evaluate_mock.py` 和 A 榜快照中的同名脚本都做以下检查：

1. 预测、真值和 noisy 都必须是有限的 `(N, 3)` 数组。
2. 预测点数必须与 noisy 点数完全一致；失败样本记 0 分并计入 `shape_errors`。
3. 预测样本集合按相对键 `shapenet/<类别>/<样本>` 对齐。缺预测不能从均值中删除，
   必须计为 0 分。
4. 精确 P2S（point-to-surface，点到面的距离）需要每个样本同时存在 mesh 和
   `norm.json`；默认缺失时直接失败。`--allow-missing-mesh-norm` 只用于调试，
   不能作为发布验收。
5. CD（Chamfer distance，倒角距离）和 P2S 都按 noisy 相对真值的改善比例转为分数，
   最终分为 `0.5 * CD_score + 0.5 * P2S_score`。这套本地实现不是平台内部实现的声明，
   只能用于本项目公开回归。
6. 显式评测清单中缺少 clean 或 noisy 输入时，评测直接失败，不能静默缩小评测集合。
   未提供清单时，clean 与 noisy 的相对键集合也必须一致。

发布前使用两个合成样本验证了 A/B 两套评测器：clean-as-pred 为 100，noisy-as-pred 为 0，
缺失或无效预测仍计入分母，缺失输入、mesh 或归一化文件会被拒绝。NumPy 与
`point_cloud_utils` 两个 P2S 后端各通过 22 个用例，见 [发布验证记录](release_preparation.md)。

CD 的实现是：以 clean 点云的 bbox（包围盒）中心和最大半径得到单位球变换，将同一
变换应用到 pred 和 noisy，再计算对称平方最近邻距离。P2S 的实现是：读取样本的
`norm.json`，用其中的 center/scale 变换 mesh，再计算点到 mesh 表面的距离；没有
`point_cloud_utils` 时会使用纯 NumPy 的点到三角面实现，数值口径不变但速度较慢；
报告会注明实际使用的 P2S 后端。若环境也没有 `trimesh`，评测器可读取基础 OBJ，
其他网格格式仍需要安装 `trimesh`。

## 4. 坐标变换与数据处理

### 4.1 在线训练链

从 mesh 动态生成训练样本时，公开 transform（变换）链的顺序是：

```text
mesh sampling -> normalize -> add Laplace noise -> optional linear -> patch
```

其中：

```text
center = (max(clean_sample) + min(clean_sample)) / 2
scale  = max_i ||clean_sample[i] - center||_2
clean_normalized = (clean_sample - center) / scale
noisy = clean_normalized + Laplace(0, sigma)
```

同一个 center/scale 必须用于 clean、noisy、辅助 dense clean 和 P2S mesh。`norm.json`
保存这两个量以及生成样本时的噪声信息，不能把每个数组分别归一化。

### 4.2 B 榜预生成训练对

B 榜公开重训先由 `generate_b_training_pairs.py` 生成 50,000 点的 `clean.npy`、
`noisy.npy` 和 `norm.json`。因此 `pdlts_light_npy_pair.yaml` 有意跳过 sample、
normalize、add_noise 和 `linear`，只保留 patch。

对已经归一化并加噪的 NPY pair（点云对）再次执行 normalize 或
add_noise 会造成双重归一化/双重加噪，训练分布就不再是记录中的 B 榜训练分布。若要
增加旋转或缩放，必须使用能作用于 sampled point cloud（采样点云）的 `linear_pc`，
并重新记录该增广，而不是误以为只改 mesh 顶点的 `linear` 已经生效。

### 4.3 推理链

predict transform 必须为空。官方 noisy 输入只有点云，没有 mesh faces，因此不能复用
validate transform；否则会触发采样阶段的 mesh 依赖。推理所需的 patch 切分、模型
前向和整云拼接由推理 system（系统流程）负责。

### 4.4 FPS+KNN 覆盖率检查

`b_board/scripts/shared/coverage_sweep.py` 用最终 `patch_denoise` 的 FPS+KNN 规则检查
每个 noisy 点是否至少被一个 patch 覆盖。它不执行网络前向，不能替代 MOCK 分数或线上
成绩；它的用途是发现 seed 数、patch 大小或拼接覆盖的工程风险。

覆盖扫描读取样本的 `noisy.npy` 和 `norm.json`，缺少归一化清单的样本会记录为
`skipped_no_norm`，不会被静默排除。平移和统一缩放不会改变该几何覆盖关系，因此扫描
只能说明“对 coverage 等价”，不能证明与模型推理的全部归一化过程等价。执行时必须显式
标注阶段，结果写入 `outputs/evals/<stage>/<eval_id>/`：

```bash
cd b_board
python scripts/shared/coverage_sweep.py \
  --stage b_final \
  --datalist starter_code/datalist/mock_full.txt \
  --mock-dir dataset/mock_test \
  --eval-id b_final_coverage_mock200
```

以下处理会引入测试信息或改变评测条件：

- 用 clean、mesh 或测试标签为预测结果单独估计归一化参数。
- 对 pred、gt、noisy 各自独立归一化。
- 对 B 榜预生成 pair 再次 normalize 或重新加噪。
- 把 `evaluate.py` 中基于 `pc_gt` bbox 的旧 P2S 口径与 `evaluate_mock.py` 的
  `norm.json` 精确 P2S 混为一个分数。
- 用 `--allow-missing-mesh-norm` 或删掉缺失预测样本来抬高均值。

## 5. 未采用实验

下表列出 B 榜主要对照与采用情况。只有定性结论的实验未附完整数值。

| 实验 | 结果 | 评测范围 | 采用情况 |
|---|---|---|---|
| A 榜模型直接迁移到 B 榜 | zero-shot 为 77.24 | B 榜 online | B 榜需要重新适配训练数据，不能只换测试集和 checkpoint |
| 同一模型零训练二遍 | mock_b 相对第一遍下降约 1.42，出现过度收缩和覆盖变差 | B 榜 mock_b | 最终使用单独训练的第二遍 specialist |
| 从零动态重加噪 | 结果显著负向 | B 榜本地实验 | 保留固定训练点云对 |
| 重加噪加旋转 | 收益不稳定，区间跨 0，P2S 仍有负向风险 | B 榜本地实验 | 不因单次正收益进入提交链 |
| specialist 训练深度 | 短 specialist 是主要收益来源；50 epochs 后仅小幅继续提升 | B 榜实验与 online | 采用深训 specialist，但继续堆深度的收益已变小 |
| score-normalized loss | 在更强 base 上边际很小 | B 榜实验 | 最终 specialist 回到普通双向 Chamfer L2 |
| 提高推理 seed_k | mock_b 几乎不变 | B 榜 mock_b | 优先保证 200 个样本完整、`verdict=green` 和打包检查 |
| specialist shape 从 2,000 增至 3,750 | 等算力 Final 只增加约 0.02，CD 几乎不动 | B 榜对照 | 更多 shape 不是当前瓶颈，保留固定 2,000 shape 路线 |
| coupling hidden 64 增至 128 | first-pass mock_b 下降；低显存修正复跑更低 | B 榜 mock_b | 关闭当前容量翻倍实例，不把它泛化为其他容量设置无效 |
| GroupNorm 小探针 | 小规模 probe 正向 | 局部 smoke | 形状不重叠的完整训练未达到预设阈值，未采用 |
| A 榜 `linear` 增广 | 它只变换 mesh vertices，且配置位于 sample 之后，对已采样点云是静默 no-op | A/B transform 源码 | 增加点云增广时使用 `linear_pc`，并把该实现限制写进配置说明 |
| MOCK P2S 坐标系 | 旧 `evaluate.py` 用 gt bbox 变换 mesh，与样本 `norm.json` 口径不完全一致 | 两份 evaluator 源码和黄金测试 | MOCK 精确 P2S 使用同一样本 `norm.json`，先过 clean/noisy 黄金样例 |
| 训练对种子 | 历史 Python `hash()` 未固定 `PYTHONHASHSEED`，旧 pair 无法逐位重建 | B 榜重训说明 | 公开脚本使用稳定 SHA-256 种子；区分“流程可复现”和“线上权重逐位相同” |

## 6. 评测范围与结果记录

各项 A 榜实验设置与结果见 [A 榜实验记录](../experiments/a_board_atlas.md)。本地评测与线上成绩分别记录：

| 评测范围 | 记录内容 | 使用范围 |
|---|---|---|
| A/B 榜 online（线上） | 榜单、提交配置、Final/CD/P2S、结果包与权重对应关系 | 对应提交的线上成绩 |
| mock20 / mock200 / mock_b（本地模拟评测） | 样本清单、归一化参数、有效预测数、缺失与点数检查、Final/CD/P2S | 比较同一评测范围内的配置，不能替代线上成绩 |
| 小规模 probe（探针实验） | 数据切分、样本量、变动配置、诊断指标和对照 | 局部实验结果，不代表完整训练或测试集结果 |

未执行线上提交的实验只记录本地结果。不同样本范围的均值分别列示，缺失预测按评测器规则处理。

## 7. 可追溯文件

- `b_board/scripts/shared/run_mock_eval.py`：A/B 通用 MOCK 驱动和评测归档。
- `b_board/scripts/shared/coverage_sweep.py`：按最终 FPS+KNN 规则记录逐样本 patch 覆盖率。
- `b_board/starter_code/evaluate_mock.py`：B 榜公开 MOCK 评测实现。
- `a_board/starter_code/evaluate_mock.py`：A 榜 MOCK 评测实现。
- `b_board/starter_code/configs/transform/_shared/pdlts_light.yaml`：mesh 动态训练链。
- `b_board/starter_code/configs/transform/_shared/pdlts_light_npy_pair.yaml`：B 榜预生成 pair 链。
- `experiments/overview.md`：实验结果与采用配置。
- `b_board/reproduction.md`：B 榜重训、推理和结果边界。
- `docs/a_to_b_changes.md`：A/B 榜算法与数据处理差异。
