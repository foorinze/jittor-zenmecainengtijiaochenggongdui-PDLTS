# 第三方来源与许可

本项目自行编写的代码和说明使用根目录 [MIT 许可证](LICENSE)。下列上游代码及其改写保留各自许可和作者声明；根目录许可不替代第三方许可。竞赛数据未随包发布，数据的使用和再分发遵循提供方规则。

## 随包实现的来源

下表的 `src/` 路径同时适用于 `b_board/starter_code/` 与 `a_board/starter_code/`。

| 来源 | 本项目范围与修改 | 上游许可 |
|---|---|---|
| [PD-LTS](https://github.com/yanbiao1/PD-LTS)，参考提交 `ee759aa0a181d3d0fa35156325589e0b1d12849f` | Light 图特征、潜空间分离、ActNorm 与单调块对照；改为 Jittor，最终可逆层使用仿射耦合，训练和推理接入竞赛框架 | [MIT，Copyright (c) 2024 Yan](LICENSES/PD-LTS-MIT.txt) |
| [Implicit Normalizing Flows](https://github.com/thu-ml/implicit-normalizing-flows)，经 PD-LTS 的求解器引用 | 单调块对照中的不动点求解；最终仿射耦合模型不使用该求解器 | [MIT，Copyright (c) 2020 Cheng Lu](LICENSES/implicit-normalizing-flows-MIT.txt) |
| [MSN Point Cloud Completion 的 EMD](https://github.com/Colin97/MSN-Point-Cloud-Completion/tree/master/emd)，经 PD-LTS 的 `metric/emd/` 引入 | `src/model/emd_jittor/emd_op.py`；Minghua Liu 的 auction（拍卖）EMD CUDA 核函数改为 Jittor 内联 CUDA，并调整中间内存管理；未用于最终模型 | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| [PointMamba](https://github.com/LMD0311/PointMamba) 的 `models/hilbert.py`；原文件署名 Xiaoyang Wu、Kaixin Xu | `src/model/pdlts_light/hilbert.py`；改为 NumPy 编码和排序，未引入完整 PointMamba 网络；最终模型不启用该对照模块 | [Apache-2.0](LICENSES/Apache-2.0.txt) |
| [numpy-hilbert-curve](https://github.com/PrincetonLIPS/numpy-hilbert-curve) | 上述 Hilbert 编码的更早来源；PointMamba 原文件明确引用该项目 | [MIT，Princeton Laboratory for Intelligent Probabilistic Systems](LICENSES/numpy-hilbert-curve-MIT.txt) |
| [trimesh](https://github.com/mikedh/trimesh) 的表面采样实现 | `src/data/utils.py::sample_surface`；增加面索引、采样随机数和多组顶点复用接口 | [MIT，Michael Dawson-Haggerty](LICENSES/trimesh-MIT.txt) |

竞赛框架入口、训练/推理接口和官方评测来自赛题提供的 starter code（初始代码）。本项目对相关接口作了适配；不把赛题规则或官方评分公式声明为本项目原创。

## 实验参考

ScoreDenoise、StraightPCF 与 Density-aware Chamfer Distance（密度感知倒角距离）的比较结果见 [实验记录](experiments/a_board_atlas.md)。引用论文或记录实验结果不表示本仓库包含其完整原始实现。

DCD 参考库 `wutong16/Density_aware_Chamfer_Distance` 在本次核对时未提供明确许可证，不能将其源代码视为 MIT 授权。两个 Jittor 改写文件、一个 NumPy 对照实现及两个测试文件不随发布包分发；精确路径见 [发布文件边界](validation/release_inventory.py)。模型拒绝启用 `dcd_official_loss_mode`，最终 A/B 配置原本均关闭此分支。早期自有 `dcd_like` 对照公式与官方 DCD 不同，保留时不声称二者等价。历史实验指标与来源摘要继续保留。

## 论文引用

PDLTS 的基础方法来自以下论文：

```bibtex
@inproceedings{mao2024denoising,
  title={Denoising Point Clouds in Latent Space via Graph Convolution and Invertible Neural Network},
  author={Mao, Aihua and Yan, Biao and Ma, Zijing and He, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5768--5777},
  year={2024}
}
```

EMD 实现还对应 Minghua Liu 等人的 *Morphing and Sampling Network for Dense Point Cloud Completion*。Hilbert 编码采用 John Skilling 的 *Programming the Hilbert Curve*（2004）所述算法。DCD 实验对应 Tong Wu 等人的 *Density-aware Chamfer Distance as a Comprehensive Metric for Point Cloud Completion*（NeurIPS 2021）。

## 核对记录

本地 PD-LTS 许可证和 EMD 许可证源文件 SHA-256 分别为：

```text
PD-LTS: 09336fd9f5ae7b4e4b84d230c71fe2ede72ab6f52c60f784eae2a62503087643
EMD:    1eb85fc97224598dad1852b5d6483bbcf0aa8608790dcc657a5a2a761ae9c8c6
```

补充核对的上游许可证 Git blob（文件对象）标识：

```text
PointMamba LICENSE:        261eeb9e9f8b2b4b0d119366dda99c6fd7d35c64
numpy-hilbert-curve:       f5d1dcdb968da7402fbc008ff53d6cceca02de6f
trimesh LICENSE.md:        d0571124dc68ee63553001f17f043d19250159cc
implicit-normalizing-flows: d33f80866f34df6dfb7483f2f4bfa6b02ea846cf
```
