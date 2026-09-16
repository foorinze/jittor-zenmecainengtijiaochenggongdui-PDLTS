"""PDLTS Light 的数据处理、训练损失与推理接口。

PDLTSLightNetwork 经 src/model/parse.py 注册为 PDLTSLight。
训练循环、优化器与运行记录由 src/system/pdlts_light.py 管理，
整云切块和拼接由本目录 denoise.py 实现。

训练时，AugmentPatch 提供 (P, M, 3) 的 pc_noisy 和 pc_clean；
P 为局部块数，M 为块内点数。中心 seed_points_t 是 clean/noisy
种子点按 t 的插值，与原 PD-LTS 推理中的纯 noisy 种子中心不同。
pc_mix 不参与本模型训练。

推理输入为 NpyLazyAsset 中 (N, 3) 的 sampled_vertices_noisy；
predict_transform.augments 为空，输入不会再次采样或加噪。

基础训练使用 Chamfer + 0.1 * 固定索引 L2。原 PD-LTS Light 的 FBM
分支有效损失为 0.1 * EMD；结构与损失对照见
experiments/overview.md。仓库另有 Jittor EMD 对照实现，
最终方案未使用它。
"""

from typing import Dict, List

import jittor as jt
import numpy as np

from ..spec import ModelSpec
from ...data.asset import Asset
from .model import PDLTSLightNetwork
from .layer import safe_knn


# Loss dict 的 key，与 configs/task/_shared/train_pdlts_light.yaml 里 loss: 字段对齐
# 基础 surrogate loss (见 module docstring): Chamfer 是主导, L2 是稳定信号.
LOSS_KEY_CHAMFER = "chamfer"
LOSS_KEY_L2 = "l2"
# centroid-anchor: centroid anchor loss，仅 target_mode != paired_idx 时启用。
# 替代 fixed_L2 的弱稳定作用。
LOSS_KEY_CENTROID_ANCHOR = "centroid_anchor"
# direction loss: masked cosine direction loss .
# 仅 components.model config 中 direction_loss_variant == "cos" 时启用.
LOSS_KEY_DIR_COS = "dir_cos"
# coverage loss: anchor-preserving coverage-hole loss .
# 仅 components.model config 中 coverage_loss_mode != "off" 时启用.
# 形式: L_coverage = mean(top-k% d_y2x) where d_y2x = ||clean_j - nn_pred(clean_j)||^2.
# task yaml 必须显式声明 `coverage: <weight>`; 若 mode!=off 但 yaml 缺权重, spec.py
# forward 会 assert.
LOSS_KEY_COVERAGE = "coverage"
# density loss: dcd_like (was named "DCD" in an early spec; downgraded 2026-05-15
# because the formula does not match Wu et al. 2021 official calc_dcd. See
# 差异表见该 loss 文件自身的 docstring.)
# Only enabled when components.model config sets dcd_like_loss_mode == "on".
# Replaces chamfer; task yaml declares `dcd_like: 1.0` instead of `chamfer: 1.0`.
LOSS_KEY_DCD_LIKE = "dcd_like"
# 历史 DCD 配置字段仅保留兼容读取；发布包不提供该实现，构造时拒绝启用。
LOSS_KEY_DCD_OFFICIAL = "dcd_official"
# auxiliary loss: vanilla Sliced Wasserstein loss (third term only).
# Only enabled when components.model config sets sw_loss_variant != "off".
# task yaml declares `sw: <weight>`; weight applied by outer system spec.py.
LOSS_KEY_SW = "sw"
# auxiliary loss: PU-Net repulsion loss (pred-internal, third term only).
# Only enabled when components.model config sets repulsion_loss_variant != "off".
# task yaml declares `repulsion: <weight>`; weight applied by outer system spec.py.
LOSS_KEY_REPULSION = "repulsion"
# auxiliary loss: Hungarian one-to-one matched L2 loss (third term only).
# Only enabled when components.model config sets hungarian_loss_variant != "off".
# task yaml declares `c5_hungarian: <weight>`; weight applied by outer system spec.py.
LOSS_KEY_C5_HUNGARIAN = "c5_hungarian"
# 定向点传输：从重复点移动到覆盖不足（high-y2x）的目标位置。
# subset Hungarian matched L2 (third term only). 独立于通用 Hungarian lane。
# Only enabled when model config sets transport_loss_mode != "off".
# task yaml declares `transport_h: <weight>`; weight applied by outer system spec.py.
LOSS_KEY_TRANSPORT_H = "transport_h"
# direction-magnitude supervision: head-level masked cosine direction loss.
# Only enabled when model config sets dh1_dir_loss_mode != "off".
# task yaml declares `dh1_dir: <weight>`; weight applied by outer system spec.py.
LOSS_KEY_DH1_DIR = "dh1_dir"
# direction-magnitude supervision: head-level magnitude loss (MSE on ||disp||).
# Only enabled when model config sets dh1_mag_loss_mode != "off".
# task yaml declares `dh1_mag: <weight>`; weight applied by outer system spec.py.
LOSS_KEY_DH1_MAG = "dh1_mag"
# dense-surface auxiliary dense surface auxiliary: weak surface guard against a denser clean
# local set. Third term only, stacked on the paired baseline (chamfer + fixed_L2
# unchanged, target_mode=paired_idx). Only enabled when model config sets
# dense_aux_loss_mode != "off". task yaml declares `dense_aux: <weight>`.
#   "x2y" = pred -> nearest dense-clean 单向（finer surface anchor，方向同 P2S，
#           低风险；这是 dense-surface 主推变体）。
#   "chamfer" = 对称 dense chamfer（含 y2x 覆盖压力，coverage-hole/blind_uniform 风险，
#           仅作对照，默认不用）。
LOSS_KEY_DENSE_AUX = "dense_aux"
# soft keep-weight V1: soft keep-weight probe. Third-term split into:
#   weighted_y2x = 覆盖半边按 keep-weight 加权
#   keep_reg     = 平均保留率 + 方差地板正则
LOSS_KEY_WEIGHTED_Y2X = "weighted_y2x"
LOSS_KEY_KEEP_REG = "keep_reg"

# candidate-select-clean candidate-select-clean stage:
#   selector_aux = clean-guided 覆盖核 BCE，监督 learned selector 的 per-candidate logit
#                  （训练期用 clean 构造 target；推理期硬选 top-M，禁止 clean 泄漏）
LOSS_KEY_SELECTOR_AUX = "selector_aux"

# 备选损失（未用于提交）：auction-EMD。
# 仅 components.model config 中 emd_loss_mode == "on" 时启用，替换 chamfer；
# task yaml 声明 `emd: 0.1`（配方 0.1×EMD.sum()，见 losses/emd_sum.py）。
LOSS_KEY_EMD = "emd"
# 备选损失（未用于提交）：UniformCD 密度比对应搜索。
# 仅 components.model config 中 uniformcd_loss_mode == "on" 时启用，替换 chamfer；
# task yaml 声明 `uniformcd: <weight>`（L2 锚保留，照常声明 `l2: 0.1`）。
LOSS_KEY_UNIFORMCD = "uniformcd"


def _hungarian_matched_l2_loss(pred, clean):
    """Hungarian one-to-one matched L2 loss (training graph).

    pred:  [B, N, 3] Jittor Var (with grad)
    clean: [B, N, 3] Jittor Var (no grad needed)

    Pipeline:
      with jt.no_grad():
        cost = pairwise squared dist [B, N, N]
        cost -> CPU numpy -> scipy LSAP per batch item -> col_idx [B, N] int64
      matched_clean = clean[batch_idx, col_idx]   # gather
      loss = mean(||pred - matched_clean||^2)     # grad flows to pred

    Returns: (loss_scalar, col_idx_np)
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    B, N, _ = pred.shape

    # 1. cost matrix on GPU (inside no_grad, no graph built)
    with jt.no_grad():
        pred_det = pred  # already inside no_grad
        clean_det = clean
        diff = pred_det.unsqueeze(2) - clean_det.unsqueeze(1)  # [B, N, N, 3]
        cost = (diff ** 2).sum(dim=-1)                          # [B, N, N]
        cost_np = np.asarray(cost.numpy())

    # 2. scipy LSAP per batch item (CPU)
    col_list = []
    for b in range(B):
        row, col = linear_sum_assignment(cost_np[b])
        col_list.append(col.astype(np.int64))
    col_idx_np = np.stack(col_list, axis=0)  # [B, N]

    # 3. CPU -> GPU index (no grad)
    col_idx = jt.array(col_idx_np)           # [B, N], int64

    # 4. gather + L2 (gradient path resumes here)
    bi = jt.arange(B).unsqueeze(-1).broadcast([B, N])
    matched_clean = clean[bi, col_idx]       # [B, N, 3]
    loss = ((pred - matched_clean) ** 2).sum(dim=-1).mean()

    return loss, col_idx_np


def _hungarian_subpatch_matched_l2_loss(pred, clean, group_size=256):
    """2.0e-a local/subpatch Hungarian: 把 (B,N,3) 按 N 方向切 group_size 组,
    每组独立做 Hungarian, loss 取均值。

    这限制 matching pool 为局部子集, 减少 global shuffle 自由度,
    迫使局部 shell 内做一对一 transport。
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    import jittor as jt

    B, N, _ = pred.shape
    assert N % group_size == 0, f"N={N} must be divisible by group_size={group_size}"
    G = N // group_size

    pred_g = pred.reshape(B * G, group_size, 3)
    clean_g = clean.reshape(B * G, group_size, 3)

    return _hungarian_matched_l2_loss(pred_g, clean_g)


def _targeted_transport_loss(pred, clean, K=64):
    """Targeted transport loss (training graph).

    机制（对应跨类 dry-run，H@K64 train-ready）：
      source = duplicate non-keeper pred（每 occupied clean bin 保留 x2y 残差最小者为
               keeper，其余为可搬冗余源），按 x2y 残差降序取 K
      target = uncovered clean（未被任何 pred 选作最近邻），按 y2x 缺口降序取 K
      子集内 Hungarian 一对一匹配，loss = mean ||pred[src] - clean[tgt_matched]||^2
    source/target 选择与 col 在 no_grad 内完成；grad 只流过被选 source 的 matched L2。

    pred:  [B, N, 3] jt Var (with grad)
    clean: [B, N, 3] jt Var (no grad)
    返回 (loss_scalar 或 None, metrics dict)。逐 batch-item 累加后取均值。
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    B, N, _ = pred.shape
    target_full = np.zeros((B, N, 3), dtype=np.float32)
    mask_full = np.zeros((B, N), dtype=np.float32)
    keff_list, tgt_pct_list, srk_list = [], [], []
    for b in range(B):
        with jt.no_grad():
            pr = pred[b].numpy()
            cn = clean[b].numpy()
        M = cn.shape[0]
        # pred -> nearest clean (x2y) 用于 keeper / duplicate 判定
        from scipy.spatial import cKDTree
        d_x2y_sq, nn = cKDTree(cn).query(pr, k=1)
        d_x2y_sq = d_x2y_sq ** 2
        keeper = {}
        for i, ci in enumerate(nn):
            ci = int(ci)
            if ci not in keeper or d_x2y_sq[i] < d_x2y_sq[keeper[ci]]:
                keeper[ci] = i
        keep_set = set(keeper.values())
        dup = np.array([i for i in range(N) if i not in keep_set], dtype=np.int64)
        if dup.size == 0:
            continue
        dup = dup[np.argsort(-d_x2y_sq[dup])]
        covered = np.zeros(M, dtype=bool); covered[nn] = True
        unc = np.where(~covered)[0]
        if unc.size == 0:
            continue
        d_y2x, _ = cKDTree(pr).query(cn, k=1)
        unc = unc[np.argsort(-d_y2x[unc])]
        Keff = int(min(K, dup.size, unc.size))
        if Keff == 0:
            continue
        src_idx = dup[:Keff]; tgt_idx = unc[:Keff]
        with jt.no_grad():
            sd = pr[src_idx][:, None, :] - cn[tgt_idx][None, :, :]
            cost = (sd ** 2).sum(-1)
        _, col = linear_sum_assignment(cost)
        target_full[b, src_idx] = cn[tgt_idx[col]].astype(np.float32)
        mask_full[b, src_idx] = 1.0
        keff_list.append(Keff)
        tgt_pct_list.append(float((d_y2x < d_y2x[tgt_idx].min()).mean() * 100))
        keep_resid = np.sqrt(d_x2y_sq[list(keep_set)]).mean() if keep_set else 1e-9
        srk_list.append(float(np.sqrt(d_x2y_sq[src_idx]).mean() / max(keep_resid, 1e-12)))
    denom = float(mask_full.sum())
    if denom <= 0:
        return None, {"transport_K_eff": 0.0, "transport_target_y2x_pct": 0.0,
                      "transport_source_resid_over_keeper": 0.0, "transport_skipped": 1.0}, None
    target_j = jt.array(target_full)
    mask_j = jt.array(mask_full)
    loss = (((pred - target_j) ** 2).sum(dim=-1) * mask_j).sum() / denom
    return loss, {"transport_K_eff": float(np.mean(keff_list)),
                  "transport_target_y2x_pct": float(np.mean(tgt_pct_list)),
                  "transport_source_resid_over_keeper": float(np.mean(srk_list)),
                  "transport_skipped": 0.0}, mask_full


def _source_downweighted_mean(per_point, src_mask_np, alpha):
    """targeted transport：对 source 点（src_mask_np==1）的 per-point 损失乘 alpha，其余权 1，
    再做归一化加权平均 sum(w·loss)/sum(w)（必须按权重和归一化，
    否则降权会顺带缩小整体量级、污染 chamfer 相对 H 的权重比）。

    per_point:  (B, N) jt.Var
    src_mask_np: (B, N) numpy 0/1（source=1）
    alpha:      source 点权重（非 source 权重恒 1）
    返回标量 jt.Var。
    """
    w_np = np.where(src_mask_np > 0.5, alpha, 1.0).astype(np.float32)
    w = jt.array(w_np)
    return (per_point * w).sum() / (w.sum() + 1e-12)



def _repulsion_loss_train(pred, k: int = 6, radius: float = 0.00956,
                          h: float = 0.00382, eps: float = 1e-12):
    """PU-Net repulsion loss, training version (graph-resident).

    pred: [B, N, 3]
    returns: scalar repulsion loss
    """
    B, N, _ = pred.shape
    _, idx = safe_knn(pred, pred, k)
    idx = idx[:, :, 1:]  # remove self
    k_actual = idx.shape[-1]

    bi = jt.arange(B).unsqueeze(-1).unsqueeze(-1).broadcast([B, N, k_actual])
    nn_pred = pred[bi, idx]

    diff = nn_pred - pred.unsqueeze(-2)
    dist2 = (diff ** 2).sum(dim=-1)
    dist2 = jt.maximum(dist2, jt.array(eps))
    dist = jt.sqrt(dist2)
    weight = jt.exp(-dist2 / (h ** 2))
    return ((radius - dist) * weight).mean()


def _sample_sw_projections(D: int = 3, K: int = 100):
    """Sample global_shared [D, K] unit-norm projection matrix for SW loss.

    All batch items share the same projections to minimise random sources.
    Per-step resample (no fixed state), consistent with PointSWD design.
    """
    theta = jt.randn(D, K)
    theta_norm = jt.sqrt(jt.sum(theta ** 2, dim=0, keepdims=True))
    return theta / (theta_norm + 1e-8)


def _sample_per_sample_projections(B: int, D: int = 3, K: int = 100):
    """Sample per_sample [B, D, K] unit-norm projection matrix for SW loss.

    Each batch item gets independent projection directions.
    Normalised along D dimension (unit vectors in R^D).
    """
    theta = jt.randn(B, D, K)
    theta_norm = jt.sqrt(jt.sum(theta ** 2, dim=1, keepdims=True))
    return theta / (theta_norm + 1e-8)


def _sw_loss_vanilla(pred, clean, theta=None, K: int = 100):
    """Vanilla sliced Wasserstein loss, global_shared [3,K] projections.

    pred, clean: [B, N, 3]  same shape
    theta: optional pre-sampled [3, K]; if None, samples new per-step.
    returns: scalar SW loss
    """
    if theta is None:
        theta = _sample_sw_projections(3, K)
    proj_pred = pred @ theta                       # [B, N, K]
    proj_clean = clean @ theta                     # [B, N, K]
    sorted_pred, _ = jt.sort(proj_pred, dim=1)
    sorted_clean, _ = jt.sort(proj_clean, dim=1)
    return ((sorted_pred - sorted_clean) ** 2).mean()


def _sw_loss_per_sample(pred, clean, theta=None, K: int = 100):
    """Vanilla SW loss, per_sample [B, 3, K] projections.

    pred, clean: [B, N, 3]
    theta: optional pre-sampled [B, 3, K]; if None, samples new per-step.
    """
    B = pred.shape[0]
    if theta is None:
        theta = _sample_per_sample_projections(B, 3, K)
    proj_pred = pred @ theta                       # [B,N,3] @ [B,3,K] = [B,N,K]
    proj_clean = clean @ theta
    sorted_pred, _ = jt.sort(proj_pred, dim=1)
    sorted_clean, _ = jt.sort(proj_clean, dim=1)
    return ((sorted_pred - sorted_clean) ** 2).mean()


def _chamfer_l2(x: jt.Var, y: jt.Var) -> jt.Var:
    """对称 Chamfer（L2 版本），per-sample mean。

    x, y: (B, N, 3)  同形状
    return: scalar （batch mean）

    内部 wrap _chamfer_l2_breakdown; 保留为向后兼容入口.
    """
    chamfer, _, _, _, _, _ = _chamfer_l2_breakdown(x, y)
    return chamfer


def _chamfer_l2_breakdown(x: jt.Var, y: jt.Var):
    """对称 Chamfer + 半边/索引/gather 拆分, 供 L2 target rescue 复用.

    x, y: (B, N, 3) 同形状

    返回:
        chamfer:  scalar (d_x2y.mean() + d_y2x.mean())
        d_x2y:    (B, N) per-point 平方距离 x_i -> NN_y(x_i)
        d_y2x:    (B, M) per-point 平方距离 y_j -> NN_x(y_j)
        idx_x2y:  (B, N) int, 每个 x_i 在 y 中的最近邻下标 (最近邻 L2 target nn_idx 复用此 idx;
                  int 索引天然 stop-grad)
        nn_y:     (B, N, 3) gather 后的 y[idx_x2y]; 最近邻 L2 target 的 l2_target 直接用此值,
                  不重复 KNN
        idx_y2x:  (B, M) int, 每个 y_j 在 x 中的最近邻下标；soft-weight gather 复用
    """
    _, idx_x2y = safe_knn(x, y, 1)  # (B, N, 1)
    _, idx_y2x = safe_knn(y, x, 1)  # (B, M, 1)
    B, N, _ = x.shape
    _, M, _ = y.shape

    bi = jt.arange(B).view(-1, 1)
    idx_x2y_BN = idx_x2y.reshape(B, N)
    nn_y = y[bi, idx_x2y_BN]                              # (B, N, 3)
    nn_x = x[bi, idx_y2x.reshape(B, M)]                   # (B, M, 3)

    d_x2y = ((x - nn_y) ** 2).sum(dim=-1)                 # (B, N)
    d_y2x = ((y - nn_x) ** 2).sum(dim=-1)                 # (B, M)
    chamfer = d_x2y.mean() + d_y2x.mean()
    idx_y2x_BM = idx_y2x.reshape(B, M)
    return chamfer, d_x2y, d_y2x, idx_x2y_BN, nn_y, idx_y2x_BM


def _dense_surface_aux_loss(pred: jt.Var, dense_clean: jt.Var, variant: str = "x2y"):
    """dense-surface auxiliary dense surface auxiliary loss (third term only).

    pred:        (B*P, M, 3)   网络输出，with grad
    dense_clean: (B*P, Md, 3)  更稠密的 clean 局部 target，no grad
                 （Md 可与 M 不同，KNN 距离不要求同点数）

    variant:
      "x2y"     = mean_i ||pred_i - NN_dense(pred_i)||^2
                  pred 每点到最近 dense-clean 点的平方距离（更精细的贴面锚）。
                  方向与 P2S/fixed_L2 一致，不引入 y2x 覆盖压力，低风险，dense-surface 主推。
      "chamfer" = x2y + mean_j ||dense_j - NN_pred(dense_j)||^2
                  对称 dense chamfer；y2x 半边是覆盖压力（coverage-hole/blind_uniform 已知
                  高风险），仅作对照变体。

    用 jt.misc.knn(query, ref, 1) 取最近邻；int 索引天然 stop-grad，dense_clean
    不参与训练图，梯度只流过 pred。
    """
    B, M, _ = pred.shape

    # x2y: 每个 pred 点 -> 最近 dense clean 点
    _, idx_p2d = safe_knn(pred, dense_clean, 1)        # (B, M, 1)
    bi = jt.arange(B).view(-1, 1)
    nn_dense = dense_clean[bi, idx_p2d.reshape(B, M)]     # (B, M, 3)
    d_x2y = ((pred - nn_dense) ** 2).sum(dim=-1)          # (B, M)
    loss_x2y = d_x2y.mean()

    if variant == "x2y":
        return loss_x2y, {"dense_aux_x2y_raw": float(loss_x2y.item())}

    if variant == "chamfer":
        Md = dense_clean.shape[1]
        _, idx_d2p = safe_knn(dense_clean, pred, 1)    # (B, Md, 1)
        nn_pred = pred[bi, idx_d2p.reshape(B, Md)]        # (B, Md, 3)
        d_y2x = ((dense_clean - nn_pred) ** 2).sum(dim=-1)  # (B, Md)
        loss_y2x = d_y2x.mean()
        loss = loss_x2y + loss_y2x
        return loss, {
            "dense_aux_x2y_raw": float(loss_x2y.item()),
            "dense_aux_y2x_raw": float(loss_y2x.item()),
        }

    raise ValueError(f"unknown dense_aux variant: {variant!r}")


def _chamfer_l2_breakdown_pairwise(x: jt.Var, y: jt.Var):
    """显式 pairwise 版本，用作数值漂移审计的对照实现。"""
    B, N, _ = x.shape
    _, M, _ = y.shape
    dist2 = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(dim=-1)

    idx_x2y_BN, _ = dist2.argmin(dim=2)
    idx_y2x_BM, _ = dist2.argmin(dim=1)

    bi = jt.arange(B).view(-1, 1)
    nn_y = y[bi, idx_x2y_BN]
    nn_x = x[bi, idx_y2x_BM]

    d_x2y = ((x - nn_y) ** 2).sum(dim=-1)
    d_y2x = ((y - nn_x) ** 2).sum(dim=-1)
    chamfer = d_x2y.mean() + d_y2x.mean()
    return chamfer, d_x2y, d_y2x, idx_x2y_BN, nn_y, idx_y2x_BM


def _fcd_beta(mode: str, epoch: int, total_epochs: int, sigma: float) -> float:
    """FCD 的 beta 调度；alpha 固定为 1.0，beta 作用在 y->x completeness 项。"""
    total_epochs = max(1, int(total_epochs))
    epoch = max(0, int(epoch))
    half_epochs = total_epochs / 2.0
    if mode == "off":
        return 1.0
    if mode == "static":
        return 2.0
    if mode == "stair":
        return 1.0 if epoch > half_epochs else 2.0
    if mode == "linear":
        return max(1.0, 2.0 - (epoch / float(total_epochs)))
    if mode == "abridged":
        if epoch > half_epochs:
            return max(1.0, 3.0 - (epoch / half_epochs))
        return 2.0
    if mode == "exp":
        import math
        sigma = max(1e-6, float(sigma))
        return 1.0 + math.exp(-epoch / sigma)
    raise AssertionError(f"unreachable fcd_chamfer_mode={mode!r}")


def _topk_mean_lastdim(x: jt.Var, k: int) -> jt.Var:
    """对最后一维取 top-k 大值的 mean. coverage-hole loss 用。

    Jittor `jt.topk` 优先; 不可用则 fallback 到 argsort + gather, 仍保持梯度通路
    (dry-run 已验证两种方式 grad 都通).

    x: (..., N) 任意 batch 维度, 最后一维做 top-k
    k: 至少 1; caller 必须传 max(1, int(N * frac))
    """
    try:
        vals, _ = jt.topk(x, k, dim=-1, largest=True)  # type: ignore[arg-type]
        return vals.mean()
    except Exception:
        # fallback: argsort descending then gather first-k cols.
        # 仅 2D batch (B, N) 路径; coverage-hole 总是 (B*P, M_clean), 满足.
        idx = jt.argsort(-x, dim=-1)
        if x.ndim != 2:
            raise NotImplementedError(
                f"_topk_mean_lastdim fallback only supports 2D, got ndim={x.ndim}"
            )
        B = x.shape[0]
        rows = jt.arange(B).unsqueeze(-1).broadcast([B, k])
        cols = idx[:, :k]
        return x[rows, cols].mean()


def _weighted_fixed_l2(
    per_point_sq: jt.Var,
    mode: str,
    topk_frac: float,
    downweight: float,
):
    """fixed-index L2 的 per-point 权重版本；权重由 detached 误差决定。"""
    if mode == "uniform":
        return per_point_sq.mean(), {}
    if mode != "suspect_topk_downweight":
        raise AssertionError(f"unsupported l2_weight_mode={mode!r}")

    N = int(per_point_sq.shape[-1])
    k = max(1, int(N * float(topk_frac)))
    ref = per_point_sq.detach()
    vals, _ = jt.topk(ref, k, dim=-1, largest=True)  # type: ignore[arg-type]
    threshold = vals[:, -1:].broadcast(ref.shape)
    suspect_mask = ref >= threshold
    ones = jt.ones_like(ref)
    weights = jt.ternary(suspect_mask, ones * float(downweight), ones)
    weight_mean_raw = weights.mean()
    weights = weights / (weight_mean_raw + 1e-8)
    l2 = (weights * per_point_sq).mean()
    metrics = {
        "l2_weight_topk_frac": float(topk_frac),
        "l2_weight_downweight": float(downweight),
        "l2_weight_raw_mean": float(weight_mean_raw.item()),
        "l2_weight_norm_mean": float(weights.mean().item()),
        "l2_weight_suspect_ratio": float(suspect_mask.float32().mean().item()),
    }
    return l2, metrics


def _masked_direction_loss(
    pred_disp: jt.Var,
    gt_disp: jt.Var,
    tau_ratio: float = 0.05,
    eps: float = 1e-8,
    variant: str = "cos",
) -> Dict[str, jt.Var]:
    """Masked cosine direction loss.

    Inputs:
        pred_disp, gt_disp: (B*P, M, 3)
    Returns dict with:
        L_dir:            scalar (反传用, 1 - cos 或 angular)
        L_mag_logonly:    scalar (仅记录, 不反传, log-ratio 的 MSE)
        disp_cos_masked:  scalar (= 1 - L_dir, 便于直读)
        disp_scale_masked:scalar (= mean(pred_norm/gt_norm)[mask])
        dir_mask_ratio:   scalar (mask 命中比例)

    Mask: gt_norm >= tau_ratio * per-patch-median(gt_norm), 且 pred_norm >= eps.
    per-patch median 用 stop_grad + numpy, tau 不回传梯度.
    """
    # (B*P, M)
    gt_norm = (gt_disp * gt_disp).sum(dim=-1).sqrt()
    pred_norm = (pred_disp * pred_disp).sum(dim=-1).sqrt()

    # per-patch median(gt_norm), 无梯度; 用 numpy 算稳
    gt_norm_np = gt_norm.stop_grad().numpy()                # (B*P, M)
    median_per_patch = np.median(gt_norm_np, axis=-1)       # (B*P,)
    tau_np = (tau_ratio * median_per_patch).astype(np.float32)
    tau = jt.array(tau_np).unsqueeze(-1)                    # (B*P, 1)

    mask = (gt_norm >= tau) & (pred_norm >= eps)            # bool (B*P, M)
    mask_f = mask.float32()
    n_valid = mask_f.sum() + eps                            # scalar

    # cosine (无梯度时也稳): dot / (|a||b| + eps)
    cos_elem = (pred_disp * gt_disp).sum(dim=-1) / (pred_norm * gt_norm + eps)
    cos_elem = cos_elem.clamp(-1.0 + 1e-6, 1.0 - 1e-6)

    if variant == "cos":
        # 1 - cos, masked mean
        dir_elem = 1.0 - cos_elem
    elif variant == "angular":
        # acos(cos) / pi, masked mean (fallback)
        dir_elem = jt.acos(cos_elem) / np.pi
    else:
        raise ValueError(f"unknown direction_loss variant: {variant!r}")

    L_dir = (dir_elem * mask_f).sum() / n_valid

    # log-ratio magnitude, log-only（只记录）
    log_ratio = jt.log(pred_norm + eps) - jt.log(gt_norm + eps)
    L_mag = ((log_ratio * log_ratio) * mask_f).sum() / n_valid

    # 观测指标
    disp_cos_masked = (cos_elem * mask_f).sum() / n_valid
    ratio = pred_norm / (gt_norm + eps)
    disp_scale_masked = (ratio * mask_f).sum() / n_valid
    dir_mask_ratio = mask_f.mean()

    return {
        "L_dir": L_dir,
        "L_mag_logonly": L_mag,
        "disp_cos_masked": disp_cos_masked,
        "disp_scale_masked": disp_scale_masked,
        "dir_mask_ratio": dir_mask_ratio,
    }


def _dh1_mag_loss(pred_disp: jt.Var, gt_disp: jt.Var,
                  variant: str = "mse", eps: float = 1e-8) -> Dict[str, jt.Var]:
    """Direction-magnitude supervision loss: supervise ||head_disp|| against ||gt_disp||.

    variant:
      "mse"      = MSE(||pred||, ||gt||)
      "relative" = mean(| ||pred|| - ||gt|| | / (||gt|| + eps))
    """
    pred_norm = (pred_disp ** 2).sum(dim=-1).sqrt()
    gt_norm = (gt_disp ** 2).sum(dim=-1).sqrt()
    if variant == "mse":
        L_mag = ((pred_norm - gt_norm) ** 2).mean()
    elif variant == "relative":
        L_mag = ((pred_norm - gt_norm).abs() / (gt_norm + eps)).mean()
    else:
        raise ValueError(f"unknown dh1_mag_loss variant: {variant!r}")
    mag_ratio = (pred_norm / (gt_norm + eps)).mean()
    return {"L_mag": L_mag, "mag_ratio": mag_ratio}


def _dh1_stable_mag_loss(
    pred_mag: jt.Var,
    gt_disp: jt.Var,
    eps: float = 1e-4,
    huber_delta: float = 1.0,
    gt_norm_threshold: float = 1e-5,
) -> Dict[str, jt.Var]:
    """Stable magnitude loss (direction-magnitude supervision).

    直接监督 head 输出的 pred_mag (network._dh1_head_mag), 不经过 pred_disp。
    log-space Huber: Huber(log(pred_mag+eps) - log(gt_mag+eps)).
    掩码只基于 gt_norm。

    这是 B2a 旧 mag loss (pred_disp norm → CUDA crash) 的稳定替身。
    """
    gt_mag = (gt_disp ** 2).sum(dim=-1).sqrt()  # (B*P, M)
    pred_mag_sq = pred_mag.squeeze(-1)           # (B*P, M)

    log_pred = jt.log(pred_mag_sq + eps)
    log_gt = jt.log(gt_mag + eps)
    diff = log_pred - log_gt

    # Huber: 0.5*diff^2 if |diff|<=delta else delta*(|diff|-0.5*delta)
    abs_diff = diff.abs()
    quadratic = 0.5 * (diff ** 2)
    linear = huber_delta * (abs_diff - 0.5 * huber_delta)
    huber = jt.ternary(abs_diff <= huber_delta, quadratic, linear)

    mask = gt_mag >= gt_norm_threshold
    mask_f = mask.float32()
    n_valid = mask_f.sum() + eps
    L_mag = (huber * mask_f).sum() / n_valid

    mag_ratio = (pred_mag_sq / (gt_mag + eps))
    mag_ratio_masked = (mag_ratio * mask_f).sum() / n_valid

    return {"L_mag": L_mag, "mag_ratio": mag_ratio_masked}


def _dh1_unit_dir_loss(
    pred_unit_dir: jt.Var,
    gt_disp: jt.Var,
    eps: float = 1e-4,
    gt_norm_threshold: float = 1e-5,
) -> Dict[str, jt.Var]:
    """Unit-direction supervision (direction-magnitude supervision).

    直接监督 head 输出的 unit_dir，不监督 pred_disp 的 cosine。
    避免 pred_disp→0 时 ||pred_disp|| 在分母导致的梯度爆炸。
    pred_unit_dir 本身已是单位向量 (||·||=1), 分母不参与 loss。
    """
    gt_norm = (gt_disp * gt_disp).sum(dim=-1).sqrt()
    gt_unit_dir = gt_disp / (gt_norm.unsqueeze(-1) + eps)

    mask = gt_norm >= gt_norm_threshold
    mask_f = mask.float32()
    n_valid = mask_f.sum() + eps

    dot = (pred_unit_dir * gt_unit_dir).sum(dim=-1)
    dot_clamped = dot.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    dir_elem = 1.0 - dot_clamped
    L_dir = (dir_elem * mask_f).sum() / n_valid
    disp_cos_masked = (dot_clamped * mask_f).sum() / n_valid
    dir_mask_ratio = mask_f.mean()

    return {
        "L_dir": L_dir,
        "disp_cos_masked": disp_cos_masked,
        "dir_mask_ratio": dir_mask_ratio,
    }


class PDLTSLight(ModelSpec):
    """PDLTS Light 在 starter_code/src/model 体系下的 ModelSpec 实现。

    Config 字段 (见 configs/model/_shared/pdlts_light.yaml):
        __target__: PDLTSLight
        pc_channel: 3
        aug_channel: 48
        n_injector: 12
        cut_channel: 24
        nflow_module: 12
        num_neighbors: 32
        coupling_hidden: 64
        mlgc_hidden: 64
        log_scale_clamp: 0.1
    """

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        if str(cfg.get("dcd_official_loss_mode", "off")) != "off":
            raise ValueError(
                "dcd_official is excluded from this release because the upstream "
                "source has no explicit license. Final A/B models use mode='off'."
            )

        self.network = PDLTSLightNetwork(
            pc_channel=cfg.get("pc_channel", 3),
            aug_channel=cfg.get("aug_channel", 48),
            n_injector=cfg.get("n_injector", 12),
            cut_channel=cfg.get("cut_channel", 24),
            nflow_module=cfg.get("nflow_module", 12),
            num_neighbors=cfg.get("num_neighbors", 32),
            mlgc_hidden=cfg.get("mlgc_hidden", 64),
            coupling_hidden=cfg.get("coupling_hidden", 64),
            log_scale_clamp=cfg.get("log_scale_clamp", 0.1),
            direction_head_mode=str(cfg.get("direction_head_mode", "off")),
            direction_head_hidden=int(cfg.get("direction_head_hidden", 64)),
            direction_head_feat_source=str(cfg.get("direction_head_feat_source", "predict_z")),
            # 全局上下文模块
            global_context_mode=str(cfg.get("global_context_mode", "off")),
            global_context_d_model=int(cfg.get("global_context_d_model", 64)),
            global_context_n_heads=int(cfg.get("global_context_n_heads", 4)),
            global_context_n_layers=int(cfg.get("global_context_n_layers", 2)),
            global_context_ffn_multiplier=int(cfg.get("global_context_ffn_multiplier", 2)),
            # GroupToken 骨干网络配置
            backbone_mode=str(cfg.get("backbone_mode", "mlgc")),
            backbone_G=int(cfg.get("backbone_G", 64)),
            backbone_S=int(cfg.get("backbone_S", 32)),
            backbone_C=int(cfg.get("backbone_C", 128)),
            backbone_depth=int(cfg.get("backbone_depth", 4)),
            backbone_heads=int(cfg.get("backbone_heads", 4)),
            backbone_upsample_k=int(cfg.get("backbone_upsample_k", 8)),
            backbone_point_id_dim=int(cfg.get("backbone_point_id_dim", 0)),
            backbone_point_id_gamma_init=float(cfg.get("backbone_point_id_gamma_init", 0.05)),
            # post-FBM residual head
            residual_head_mode=str(cfg.get("residual_head_mode", "off")),
            residual_head_hidden=int(cfg.get("residual_head_hidden", 64)),
            # soft keep-weight V1: keep-weight head
            keep_head_mode=str(cfg.get("keep_head_mode", "off")),
            keep_head_hidden=int(cfg.get("keep_head_hidden", 64)),
            # 候选生成、选择与精修
            candidate_mode=str(cfg.get("candidate_mode", "off")),
            candidate_R=int(cfg.get("candidate_R", 4)),
            candidate_knn_k=int(cfg.get("candidate_knn_k", 6)),
            selector_mode=str(cfg.get("selector_mode", "off")),
            selector_hidden=int(cfg.get("selector_hidden", 64)),
            cleaner_mode=str(cfg.get("cleaner_mode", "off")),
            cleaner_hidden=int(cfg.get("cleaner_hidden", 64)),
            # confidence-gated pull head, default off.
            pull_head_mode=str(cfg.get("pull_head_mode", "off")),
            pull_head_hidden=int(cfg.get("pull_head_hidden", 64)),
            pull_delta_max=float(cfg.get("pull_delta_max", 0.0)),
            pull_head_feat_source=str(cfg.get("pull_head_feat_source", "predict_z")),
        )

        # centroid-anchor: target_mode 控制训练 target 来源。
        # "paired_idx"                  = 1.x 默认行为
        # "clean_knn_noisy_seed_nn"     = clean KNN around the nearest clean point
        self.target_mode: str = str(cfg.get("target_mode", "paired_idx"))
        assert self.target_mode in ("paired_idx", "clean_knn_noisy_seed_nn"), (
            f"target_mode must be paired_idx or clean_knn_noisy_seed_nn, "
            f"got {self.target_mode!r}"
        )
        # centroid anchor 权重，仅 target_mode != paired_idx 时生效
        self.centroid_anchor_weight: float = float(cfg.get("centroid_anchor_weight", 0.01))

        # direction-loss 开关 .
        # "off" = paired baseline行为, 只返 chamfer + l2;
        # "cos" = v1, masked cosine direction loss 进 loss_dict[LOSS_KEY_DIR_COS];
        # "angular" = fallback (§6 允许一次), 同样进 LOSS_KEY_DIR_COS (key 不变, variant 改).
        self.direction_loss_variant: str = str(cfg.get("direction_loss_variant", "off"))
        self.direction_mask_tau_ratio: float = float(cfg.get("direction_mask_tau_ratio", 0.05))
        assert self.direction_loss_variant in ("off", "cos", "angular"), (
            f"direction_loss_variant must be off/cos/angular, got {self.direction_loss_variant!r}"
        )

        # L2 target 开关.
        # "fixed"         = paired baseline行为, l2 = ((denoised - pc_clean_flat)**2).sum(-1).mean()
        # "off"           = l2 完全不参与, loss_dict 不含 'l2' key (spec.py forward 自然跳过)
        # "pred_nn_clean" = epoch < warmup 用 fixed; epoch >= warmup 用 chamfer idx_x2y gather
        #
        # YAML 防御: YAML 1.1 把裸 `off/on/yes/no` 解析成 bool (Norway problem).
        # 在 yaml 里必须写 `l2_target_mode: "off"` (带引号); 若漏了引号会被读成 False,
        # 下面类型检查会给出诊断级错误.
        _raw_mode = cfg.get("l2_target_mode", "fixed")
        if isinstance(_raw_mode, bool):
            raise AssertionError(
                f"l2_target_mode was parsed as bool {_raw_mode!r}; "
                f"YAML likely interpreted unquoted off/on/yes/no as boolean. "
                f"Use quoted form in yaml: `l2_target_mode: \"off\"`"
            )
        self.l2_target_mode: str = str(_raw_mode)
        self.l2_target_warmup_epochs: int = int(cfg.get("l2_target_warmup_epochs", 0))
        assert self.l2_target_mode in ("fixed", "off", "pred_nn_clean"), (
            f"l2_target_mode must be fixed/off/pred_nn_clean, got {self.l2_target_mode!r}"
        )
        if self.l2_target_mode == "off" and self.l2_target_warmup_epochs != 0:
            import warnings as _warnings
            _warnings.warn(
                f"l2_target_mode=off ignores l2_target_warmup_epochs (got "
                f"{self.l2_target_warmup_epochs}); warmup has no meaning when L2 is absent.",
                RuntimeWarning,
            )
        self.l2_weight_mode: str = str(cfg.get("l2_weight_mode", "uniform"))
        self.l2_weight_topk_frac: float = float(cfg.get("l2_weight_topk_frac", 0.2))
        self.l2_weight_downweight: float = float(cfg.get("l2_weight_downweight", 0.5))
        assert self.l2_weight_mode in ("uniform", "suspect_topk_downweight"), (
            f"l2_weight_mode must be uniform/suspect_topk_downweight, got {self.l2_weight_mode!r}"
        )
        assert 0.0 < self.l2_weight_topk_frac <= 1.0, (
            f"l2_weight_topk_frac must be in (0, 1], got {self.l2_weight_topk_frac}"
        )
        assert 0.0 < self.l2_weight_downweight <= 1.0, (
            f"l2_weight_downweight must be in (0, 1], got {self.l2_weight_downweight}"
        )
        if self.l2_weight_mode != "uniform":
            assert self.l2_target_mode == "fixed", (
                "weighted L2 mode l2_weight_mode only changes fixed-index L2 weights; "
                "keep l2_target_mode='fixed' to avoid mixing target and weight axes"
            )

        # 当前训练 epoch (由 outer system on_train_epoch_start 通过 set_train_epoch 设置).
        # 默认 0 让非训练场景 (eval / smoke dry-run) 也能运行; pred_nn_clean warmup
        # 逻辑 `epoch < warmup_epochs` 在默认 0 下等同 "还没过 warmup".
        self._current_train_epoch: int = 0

        # coverage-hole loss : anchor-preserving coverage
        # hole loss. "off" = 不启用, loss_dict 不含 coverage key, 不需要 task yaml
        # 声明 coverage 权重; "hole_top10" = L_coverage = mean(top-k% d_y2x).
        # 默认 off 保持所有旧 config 行为不变.
        self.coverage_loss_mode: str = str(cfg.get("coverage_loss_mode", "off"))
        assert self.coverage_loss_mode in ("off", "hole_top10"), (
            f"coverage_loss_mode must be off/hole_top10, got {self.coverage_loss_mode!r}"
        )
        self.coverage_top_k_frac: float = float(cfg.get("coverage_top_k_frac", 0.1))
        assert 0.0 < self.coverage_top_k_frac <= 1.0, (
            f"coverage_top_k_frac must be in (0, 1], got {self.coverage_top_k_frac}"
        )

        # dcd_like loss (originally named "DCD probe"; downgraded
        # 公式差异见 losses/dcd_like.py).
        # "off" = disabled, use chamfer; "on" = replace chamfer, loss_dict
        # contains 'dcd_like' instead of 'chamfer'.
        self.dcd_like_loss_mode: str = str(cfg.get("dcd_like_loss_mode", "off"))
        assert self.dcd_like_loss_mode in ("off", "on"), (
            f"dcd_like_loss_mode must be off/on, got {self.dcd_like_loss_mode!r}"
        )
        self.dcd_like_alpha: float = float(cfg.get("dcd_like_alpha", 1000.0))
        self.dcd_like_n_lambda: float = float(cfg.get("dcd_like_n_lambda", 0.5))

        # 读取历史配置的关闭状态；启用已在构造入口被拒绝。
        self.dcd_official_loss_mode: str = str(cfg.get("dcd_official_loss_mode", "off"))
        assert self.dcd_official_loss_mode in ("off", "on"), (
            f"dcd_official_loss_mode must be off/on, got {self.dcd_official_loss_mode!r}"
        )
        if self.dcd_official_loss_mode == "on":
            assert self.dcd_like_loss_mode == "off", (
                "dcd_like and dcd_official are mutually exclusive; "
                "set dcd_like_loss_mode='off' in the model yaml"
            )
            assert "dcd_official_alpha" in cfg, (
                "dcd_official_loss_mode='on' requires explicit dcd_official_alpha "
                "in the model yaml (no default; pick from the scale-sanity dry-run verdict)"
            )
            assert "dcd_official_n_lambda" in cfg, (
                "dcd_official_loss_mode='on' requires explicit dcd_official_n_lambda "
                "in the model yaml (no default; paper experiments use 0.5)"
            )
        self.dcd_official_alpha: float = float(cfg.get("dcd_official_alpha", 0.0))
        self.dcd_official_n_lambda: float = float(cfg.get("dcd_official_n_lambda", 0.0))

        # emd / uniformcd：替换 chamfer 的两个互斥分支（均未用于提交），
        # 与 dcd 两 lane 共同构成"四选一"（默认全 off = 使用默认 Chamfer 损失）。

        self.emd_loss_mode: str = str(cfg.get("emd_loss_mode", "off"))
        assert self.emd_loss_mode in ("off", "on"), (
            f"emd_loss_mode must be off/on, got {self.emd_loss_mode!r}"
        )
        self.uniformcd_loss_mode: str = str(cfg.get("uniformcd_loss_mode", "off"))
        assert self.uniformcd_loss_mode in ("off", "on"), (
            f"uniformcd_loss_mode must be off/on, got {self.uniformcd_loss_mode!r}"
        )
        _replace_chamfer_lanes = (
            self.dcd_like_loss_mode, self.dcd_official_loss_mode,
            self.emd_loss_mode, self.uniformcd_loss_mode,
        )
        assert sum(m == "on" for m in _replace_chamfer_lanes) <= 1, (
            "dcd_like/dcd_official/emd/uniformcd 四 lane 互斥（最多一个 on），got "
            f"{_replace_chamfer_lanes!r}"
        )
        self.emd_eps: float = float(cfg.get("emd_eps", 0.005))
        self.emd_iters: int = int(cfg.get("emd_iters", 50))
        self.uniformcd_k: int = int(cfg.get("uniformcd_k", 8))

        # UniformCD 也可作为 Chamfer 之上的附加项（未用于提交）
        # （不替换 chamfer，与替换 lane 互斥；系数由 task yaml `uniformcd: 0.15` 施加）。

        self.uniformcd_nudge_mode: str = str(cfg.get("uniformcd_nudge_mode", "off"))
        assert self.uniformcd_nudge_mode in ("off", "on"), (
            f"uniformcd_nudge_mode must be off/on, got {self.uniformcd_nudge_mode!r}"
        )
        if self.uniformcd_nudge_mode == "on":
            assert all(m == "off" for m in _replace_chamfer_lanes), (
                "uniformcd_nudge_mode=on requires the base Chamfer loss; "
                f"set dcd_like/dcd_official/emd/uniformcd to off, got {_replace_chamfer_lanes!r}"
            )
        # r 裁剪区间，建议 [0.5,2.0]；默认 None = 不裁剪。
        _ucd_clip_lo = cfg.get("uniformcd_clip_lo", None)
        _ucd_clip_hi = cfg.get("uniformcd_clip_hi", None)
        self.uniformcd_clip_lo = None if _ucd_clip_lo is None else float(_ucd_clip_lo)
        self.uniformcd_clip_hi = None if _ucd_clip_hi is None else float(_ucd_clip_hi)
        # 计算后端：默认 numpy；
        # jittor = 密度/kNN/argmin 全程 GPU（数学式一致，f32 vs f64 微差）。
        self.uniformcd_backend: str = str(cfg.get("uniformcd_backend", "numpy"))
        assert self.uniformcd_backend in ("numpy", "jittor"), (
            f"uniformcd_backend must be numpy/jittor, got {self.uniformcd_backend!r}"
        )

        # auxiliary-loss 漂移审计控制开关。默认保持当前行为不变。
        self.chamfer_impl: str = str(cfg.get("chamfer_impl", "knn_breakdown"))
        assert self.chamfer_impl in ("knn_breakdown", "pairwise_legacy_control"), (
            f"chamfer_impl must be knn_breakdown/pairwise_legacy_control, got {self.chamfer_impl!r}"
        )
        self.train_r3_metrics: bool = bool(cfg.get("train_r3_metrics", True))

        # completeness weighting FCD schedule。
        # 官方 FCD 是 PyTorch/CUDA extension；这里仅复写权重调度语义：
        # chamfer = mean(d_x2y) + beta(epoch) * mean(d_y2x)。默认 off 完全等价旧 Chamfer。
        self.fcd_chamfer_mode: str = str(cfg.get("fcd_chamfer_mode", "off"))
        self.fcd_total_epochs: int = int(cfg.get("fcd_total_epochs", 100))
        self.fcd_exp_sigma: float = float(cfg.get("fcd_exp_sigma", float(self.fcd_total_epochs) / 2.0))
        assert self.fcd_chamfer_mode in ("off", "static", "stair", "linear", "abridged", "exp"), (
            "fcd_chamfer_mode must be off/static/stair/linear/abridged/exp, "
            f"got {self.fcd_chamfer_mode!r}"
        )
        assert self.fcd_total_epochs > 0, (
            f"fcd_total_epochs must be positive, got {self.fcd_total_epochs}"
        )

        # auxiliary-loss SW probe（小规模 dry-run 已验证，λ=0.5 已校准）。
        # "off" = disabled (default, all existing configs unchanged).
        # "vanilla_global_shared" = global [3,K] per-step resample.
        self.sw_loss_variant: str = str(cfg.get("sw_loss_variant", "off"))
        self.sw_num_projections: int = int(cfg.get("sw_num_projections", 100))
        self.sw_projection_scope: str = str(cfg.get("sw_projection_scope", "global_shared"))
        self.sw_projection_schedule: str = str(cfg.get("sw_projection_schedule", "per_step"))
        assert self.sw_loss_variant in ("off", "vanilla_global_shared", "vanilla_per_sample"), (
            f"sw_loss_variant must be off/vanilla_global_shared/vanilla_per_sample, got {self.sw_loss_variant!r}"
        )
        assert self.sw_projection_scope in ("global_shared", "per_sample"), (
            f"sw_projection_scope must be global_shared/per_sample, got {self.sw_projection_scope!r}"
        )
        assert self.sw_projection_schedule in ("per_step", "per_epoch"), (
            f"sw_projection_schedule must be per_step/per_epoch, got {self.sw_projection_schedule!r}"
        )
        # Cached projection for per_epoch schedule (regenerated on epoch boundary).
        self._cached_sw_theta = None

        # auxiliary-loss repulsion probe（小规模 dry-run 已校准）。
        # "off" = disabled (default).
        # "punet" = PU-Net repulsion: mean((radius - dist) * exp(-dist²/h²)).
        self.repulsion_loss_variant: str = str(cfg.get("repulsion_loss_variant", "off"))
        self.repulsion_k: int = int(cfg.get("repulsion_k", 6))
        self.repulsion_radius: float = float(cfg.get("repulsion_radius", 0.00956))
        self.repulsion_h: float = float(cfg.get("repulsion_h", 0.00382))
        assert self.repulsion_loss_variant in ("off", "punet"), (
            f"repulsion_loss_variant must be off/punet, got {self.repulsion_loss_variant!r}"
        )

        # auxiliary-loss Hungarian one-to-one assignment probe (dry-run 2026-05-20).
        # "off" = disabled (default).
        # "vanilla" = full N=1024 Hungarian + matched L2 loss.
        # "local_subpatch" = 2.0e-a: 把 patch 按 seed distance 径向分组, 每组独立 Hungarian.
        self.hungarian_loss_variant: str = str(cfg.get("hungarian_loss_variant", "off"))
        self.hungarian_n_eval: int = int(cfg.get("hungarian_n_eval", 1024))
        self.hungarian_subpatch_size: int = int(cfg.get("hungarian_subpatch_size", 256))
        assert self.hungarian_loss_variant in ("off", "vanilla", "local_subpatch"), (
            f"hungarian_loss_variant must be off/vanilla/local_subpatch, "
            f"got {self.hungarian_loss_variant!r}"
        )
        if self.hungarian_loss_variant == "local_subpatch":
            assert self.hungarian_n_eval >= 1024, (
                f"local_subpatch requires hungarian_n_eval >= patch_size (1024), "
                f"got {self.hungarian_n_eval}"
            )
        else:
            assert self.hungarian_n_eval in (128, 256, 512, 1024), (
                f"hungarian_n_eval must be 128/256/512/1024, got {self.hungarian_n_eval}"
            )

        # targeted transport (cross-class dry-run train-ready @ K=64).
        # "off" = disabled (default). "targeted_h" = duplicate src → uncovered target subset Hungarian.
        # "targeted_h_v3" = targeted transport: 同 targeted_h，但对被选 source 点的 chamfer-x2y 与 fixed-L2
        #   部分降权（解除「base 把 source 拉回原最近 clean」的拔河），y2x 半边不动。
        self.transport_loss_mode: str = str(cfg.get("transport_loss_mode", "off"))
        self.transport_k: int = int(cfg.get("transport_k", 64))
        # targeted transport source-point base downweight 系数（仅 targeted_h_v3 生效；1.0=不降权）。
        self.transport_source_x2y_alpha: float = float(
            cfg.get("transport_source_x2y_alpha", 0.2))
        self.transport_source_l2_alpha: float = float(
            cfg.get("transport_source_l2_alpha", 0.2))
        assert self.transport_loss_mode in ("off", "targeted_h", "targeted_h_v3"), (
            f"transport_loss_mode must be off/targeted_h/targeted_h_v3, "
            f"got {self.transport_loss_mode!r}"
        )
        if self.transport_loss_mode in ("targeted_h", "targeted_h_v3"):
            assert 8 <= self.transport_k <= 512, (
                f"transport_k must be in [8,512], got {self.transport_k}"
            )
            assert 0.0 <= self.transport_source_x2y_alpha <= 1.0, (
                f"transport_source_x2y_alpha must be in [0,1], "
                f"got {self.transport_source_x2y_alpha}"
            )
            assert 0.0 <= self.transport_source_l2_alpha <= 1.0, (
                f"transport_source_l2_alpha must be in [0,1], "
                f"got {self.transport_source_l2_alpha}"
            )

        # direction-magnitude supervision probe: head-level direction + magnitude loss.
        # "off" = disabled (default, B1 behavior).
        # dh1_dir_loss_mode = "masked_cos" → masked cosine loss on head displacement.
        # dh1_mag_loss_mode = "mse" | "relative" → magnitude supervision.
        self.dh1_dir_loss_mode: str = str(cfg.get("dh1_dir_loss_mode", "off"))
        self.dh1_mag_loss_mode: str = str(cfg.get("dh1_mag_loss_mode", "off"))
        self.dh1_dir_tau_ratio: float = float(cfg.get("dh1_dir_tau_ratio", 0.05))
        # direction target mode. "fixed" = clean_i - noisy_i; "nn_clean" = nearest_clean - noisy_i.
        self.dh1_dir_target_mode: str = str(cfg.get("dh1_dir_target_mode", "fixed"))
        assert self.dh1_dir_target_mode in ("fixed", "nn_clean"), (
            f"dh1_dir_target_mode must be fixed/nn_clean, got {self.dh1_dir_target_mode!r}"
        )
        assert self.dh1_dir_loss_mode in ("off", "masked_cos"), (
            f"dh1_dir_loss_mode must be off/masked_cos, got {self.dh1_dir_loss_mode!r}"
        )
        assert self.dh1_mag_loss_mode in ("off", "mse", "relative", "stable_huber_log"), (
            f"dh1_mag_loss_mode must be off/mse/relative/stable_huber_log, "
            f"got {self.dh1_mag_loss_mode!r}"
        )

        # dense-surface auxiliary dense surface auxiliary (third term only, stacked on paired baseline).
        # "off" = disabled (default; all existing configs unchanged).
        # "x2y" = pred -> nearest dense-clean 单向 surface anchor（主推，低风险）。
        # "chamfer" = 对称 dense chamfer（含 y2x 覆盖压力，仅对照）。
        # 仅 mode != off 时把 LOSS_KEY_DENSE_AUX 放进 loss_dict；需要 transform 端
        # AugmentPatch.dense_target_size > 0 提供 pc_clean_dense_local。
        self.dense_aux_loss_mode: str = str(cfg.get("dense_aux_loss_mode", "off"))
        assert self.dense_aux_loss_mode in ("off", "x2y", "chamfer"), (
            f"dense_aux_loss_mode must be off/x2y/chamfer, got {self.dense_aux_loss_mode!r}"
        )
        if self.dense_aux_loss_mode != "off":
            # dense-surface 配置限制：dense aux 使用配对数据，并保留固定索引 L2，
            # 不与其它第三项同开（防混轴）。照 transport guard 模式完整 enforce。
            assert self.target_mode == "paired_idx", (
                f"dense_aux requires target_mode='paired_idx' (paired baseline), "
                f"got {self.target_mode!r}"
            )
            assert self.l2_target_mode == "fixed", (
                f"dense_aux requires l2_target_mode='fixed' (fixed_L2 anchor 不能动), "
                f"got {self.l2_target_mode!r}"
            )
            assert self.direction_loss_variant == "off", (
                f"dense_aux cannot coexist with direction_loss_variant="
                f"{self.direction_loss_variant!r} (防混轴)"
            )
            assert self.coverage_loss_mode == "off", (
                f"dense_aux cannot coexist with coverage_loss_mode="
                f"{self.coverage_loss_mode!r}"
            )
            assert self.sw_loss_variant == "off", (
                f"dense_aux cannot coexist with sw_loss_variant={self.sw_loss_variant!r}"
            )
            assert self.repulsion_loss_variant == "off", (
                f"dense_aux cannot coexist with repulsion_loss_variant="
                f"{self.repulsion_loss_variant!r}"
            )
            assert self.hungarian_loss_variant == "off", (
                f"dense_aux cannot coexist with hungarian_loss_variant="
                f"{self.hungarian_loss_variant!r}"
            )
            assert self.transport_loss_mode == "off", (
                f"dense_aux cannot coexist with transport_loss_mode="
                f"{self.transport_loss_mode!r}"
            )
            assert self.dh1_dir_loss_mode == "off" and self.dh1_mag_loss_mode == "off", (
                f"dense_aux cannot coexist with DH1 losses (dir="
                f"{self.dh1_dir_loss_mode!r}, mag={self.dh1_mag_loss_mode!r})"
            )
            assert self.dcd_like_loss_mode == "off" and self.dcd_official_loss_mode == "off", (
                f"dense_aux cannot coexist with DCD losses (dcd_like="
                f"{self.dcd_like_loss_mode!r}, dcd_official={self.dcd_official_loss_mode!r}); "
                f"DCD replaces chamfer 与 dense aux 架在 chamfer base 的前提冲突"
            )

        # soft keep-weight. 配置限制：只允许 paired base
        # + keep_head/weighted_y2x，不与其它第三项或方向头监督混轴。
        self.keep_loss_mode: str = str(cfg.get("keep_loss_mode", "off"))
        self.keep_target_keep: float = float(cfg.get("keep_target_keep", 0.67))
        self.keep_var_floor: float = float(cfg.get("keep_var_floor", 0.01))
        self.keep_var_lambda: float = float(cfg.get("keep_var_lambda", 1.0))
        # candidate-select-clean: slot-variant selector 辅助 loss 开关（off | variant_ce）
        self.selector_aux_mode: str = str(cfg.get("selector_aux_mode", "off"))
        assert self.selector_aux_mode in ("off", "variant_ce"), (
            f"selector_aux_mode must be off/variant_ce, got {self.selector_aux_mode!r}"
        )
        assert self.keep_loss_mode in ("off", "soft_weight"), (
            f"keep_loss_mode must be off/soft_weight, got {self.keep_loss_mode!r}"
        )
        if self.keep_loss_mode != "off":
            assert getattr(self.network, "keep_head_mode", "off") == "soft_weight", (
                "keep_loss_mode requires keep_head_mode='soft_weight'"
            )
            assert self.target_mode == "paired_idx", (
                f"keep_loss_mode requires target_mode='paired_idx', got {self.target_mode!r}"
            )
            assert self.l2_target_mode == "fixed", (
                f"keep_loss_mode requires l2_target_mode='fixed', got {self.l2_target_mode!r}"
            )
            assert self.direction_loss_variant == "off", (
                f"keep_loss_mode cannot coexist with direction_loss_variant={self.direction_loss_variant!r}"
            )
            assert self.coverage_loss_mode == "off", (
                f"keep_loss_mode cannot coexist with coverage_loss_mode={self.coverage_loss_mode!r}"
            )
            assert self.sw_loss_variant == "off", (
                f"keep_loss_mode cannot coexist with sw_loss_variant={self.sw_loss_variant!r}"
            )
            assert self.repulsion_loss_variant == "off", (
                f"keep_loss_mode cannot coexist with repulsion_loss_variant={self.repulsion_loss_variant!r}"
            )
            assert self.hungarian_loss_variant == "off", (
                f"keep_loss_mode cannot coexist with hungarian_loss_variant={self.hungarian_loss_variant!r}"
            )
            assert self.transport_loss_mode == "off", (
                f"keep_loss_mode cannot coexist with transport_loss_mode={self.transport_loss_mode!r}"
            )
            assert self.dh1_dir_loss_mode == "off" and self.dh1_mag_loss_mode == "off", (
                f"keep_loss_mode cannot coexist with DH1 losses (dir={self.dh1_dir_loss_mode!r}, "
                f"mag={self.dh1_mag_loss_mode!r})"
            )
            assert self.dense_aux_loss_mode == "off", (
                "keep_loss_mode cannot coexist with dense_aux_loss_mode "
                f"{self.dense_aux_loss_mode!r}"
            )
            assert self.dcd_like_loss_mode == "off" and self.dcd_official_loss_mode == "off", (
                f"keep_loss_mode cannot coexist with DCD losses (dcd_like="
                f"{self.dcd_like_loss_mode!r}, dcd_official={self.dcd_official_loss_mode!r})"
            )
            assert 0.0 < self.keep_target_keep < 1.0, (
                f"keep_target_keep must be in (0,1), got {self.keep_target_keep}"
            )
            assert self.keep_var_floor >= 0.0, (
                f"keep_var_floor must be >=0, got {self.keep_var_floor}"
            )
            assert self.keep_var_lambda >= 0.0, (
                f"keep_var_lambda must be >=0, got {self.keep_var_lambda}"
            )

        # System 层读取此属性写 epoch_summary.jsonl（日志扩展）。
        # 推理/validate 都不写; 仅 training_step 覆盖.
        self._last_train_metrics: Dict[str, float] = {}

    def set_train_epoch(self, epoch: int) -> None:
        """由 outer system (src/system/pdlts_light.py::on_train_epoch_start) 调用,
        告知 model 当前训练 epoch, 供 pred_nn_clean warmup 判断使用.

        epoch 来源: epoch 不能在 model 内自维护 (forward 不知道自己
        是训练第几个 epoch), 必须由 outer system 传入.

        约定: 传入的是 **absolute epoch** (start_epoch + current_epoch), 与 ckpt
        命名 abs_ep 同口径; 这样 resume 训练时 warmup 不会重置.
        """
        self._current_train_epoch = int(epoch)
        # Invalidate cached SW projection on epoch boundary for per_epoch schedule.
        if self.sw_projection_schedule == "per_epoch" and self.sw_loss_variant != "off":
            self._cached_sw_theta = None

    # ---------- 数据契约 ----------
    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        """把 Dataset 出来的 Asset list 转成 tensor dict list。

        训练 / 验证时：读 asset.meta['pc_noisy' / 'pc_clean']，
        推理时：读 asset.sampled_vertices_noisy。
        """
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None, (
                    "training/validation requires asset.meta with pc_noisy/pc_clean; "
                    "check transform pipeline includes AugmentPatch"
                )
                # (P, M, 3) - P=num_patches, M=patch_size
                d = {
                    "pc_noisy": b.meta["pc_noisy"].astype(np.float32),
                    "pc_clean": b.meta["pc_clean"].astype(np.float32),
                }
                # centroid-anchor: 如果 transform 生成了 pc_clean_target，透传
                if "pc_clean_target" in b.meta:
                    d["pc_clean_target"] = b.meta["pc_clean_target"].astype(np.float32)
                # dense-surface auxiliary: dense surface auxiliary local target 透传
                if "pc_clean_dense_local" in b.meta:
                    d["pc_clean_dense_local"] = b.meta["pc_clean_dense_local"].astype(np.float32)
            else:
                # 推理走 NpyLazyAsset -> sampled_vertices_noisy (N, 3)
                d = {
                    "pc_noisy": b.sampled_vertices_noisy.astype(np.float32),
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices.astype(np.float32)
            res.append(d)
        return res

    # ---------- 训练 ----------
    def training_step(self, batch: Dict) -> Dict:
        """从 batched patch dict 算 loss。

        batch 字段 (来自 PCDataset._collate_fn 的 stack):
            pc_noisy: (B, P, M, 3)
            pc_clean: (B, P, M, 3)

        返回 loss 字典，key 必须与 configs/task/_shared/train_pdlts_light.yaml 的 loss: 字段对齐。

        方向损失 : 当 model_config.direction_loss_variant != "off"
        时额外返回 LOSS_KEY_DIR_COS, 并把 per-step raw metrics stash 到
        self._last_train_metrics (供 system 层 on_train_epoch_end 写 epoch_summary.jsonl).

        L2 target rescue: 当 l2_target_mode != "fixed" 时改变
        LOSS_KEY_L2 的 target:
            "off"           : loss_dict 不含 'l2' key; spec.py forward 自然跳过
            "pred_nn_clean" : l2 target = pc_clean_flat[idx_x2y]; idx_x2y 复用
                              chamfer 内部的 pred->clean NN 下标 (int, stop-grad);
                              epoch < l2_target_warmup_epochs 时仍走 fixed.
        """
        pc_noisy = batch["pc_noisy"]                     # (B, P, M, 3)
        pc_clean = batch["pc_clean"]
        patch_size = pc_noisy.shape[-2]
        # 合并 (B, P) -> 1 维
        pc_noisy_flat = pc_noisy.reshape(-1, patch_size, 3)  # (B*P, M, 3)

        # centroid-anchor: 按 target_mode 选择 clean target 来源
        if self.target_mode == "clean_knn_noisy_seed_nn":
            if "pc_clean_target" not in batch:
                raise RuntimeError(
                    f"target_mode={self.target_mode!r} but batch has no "
                    f"'pc_clean_target'; check transform config target_mode"
                )
            pc_clean_flat = batch["pc_clean_target"].reshape(-1, patch_size, 3)
        else:
            pc_clean_flat = pc_clean.reshape(-1, patch_size, 3)

        # centroid-anchor: unpaired target 下必须禁用的 loss 守卫
        if self.target_mode != "paired_idx":
            if self.l2_target_mode != "off":
                raise ValueError(
                    f"target_mode={self.target_mode!r} requires l2_target_mode='off', "
                    f"got {self.l2_target_mode!r}"
                )
            if self.direction_loss_variant != "off":
                raise ValueError(
                    f"direction_loss_variant={self.direction_loss_variant!r} "
                    f"incompatible with target_mode={self.target_mode!r}"
                )
            if self.dh1_dir_loss_mode != "off" or self.dh1_mag_loss_mode != "off":
                raise ValueError(
                    f"DH1 loss incompatible with target_mode={self.target_mode!r}"
                )

        # targeted transport 配置检查：必须 paired base + anchor 完整，
        # 且不与其它第三项同开（防混轴）。
        if self.transport_loss_mode in ("targeted_h", "targeted_h_v3"):
            if self.target_mode != "paired_idx":
                raise ValueError(
                    f"transport_loss_mode=targeted_h requires target_mode='paired_idx', "
                    f"got {self.target_mode!r}"
                )
            if self.l2_target_mode != "fixed":
                raise ValueError(
                    f"transport_loss_mode=targeted_h requires l2_target_mode='fixed' "
                    f"(fixed_L2 anchor 不能动), got {self.l2_target_mode!r}"
                )
            if self.hungarian_loss_variant != "off":
                raise ValueError(
                    "transport_loss_mode=targeted_h cannot coexist with "
                    f"hungarian_loss_variant={self.hungarian_loss_variant!r} (use one only)"
                )
            if self.coverage_loss_mode != "off":
                raise ValueError(
                    "transport_loss_mode=targeted_h cannot coexist with "
                    f"coverage_loss_mode={self.coverage_loss_mode!r}"
                )
            if self.centroid_anchor_weight not in (0, 0.0):
                raise ValueError(
                    "transport_loss_mode=targeted_h requires centroid_anchor_weight=0, "
                    f"got {self.centroid_anchor_weight!r}"
                )

        denoised, _ldj, _loss_d = self.network(pc_noisy_flat)

        # Chamfer 拆半: 既得总和, 也得 d_x2y/d_y2x/idx/nn_y.
        # 最近邻 L2 target 的 nn target 复用 idx_x2y, 不独立再算 knn.
        if self.chamfer_impl == "pairwise_legacy_control":
            chamfer, d_x2y, d_y2x, idx_x2y_BN, nn_y, idx_y2x_BM = _chamfer_l2_breakdown_pairwise(
                denoised, pc_clean_flat
            )
        else:
            chamfer, d_x2y, d_y2x, idx_x2y_BN, nn_y, idx_y2x_BM = _chamfer_l2_breakdown(
                denoised, pc_clean_flat
            )

        out = {}
        epoch = self._current_train_epoch

        # FCD 基线值：在所有分支外定义，保证 dcd 分支下 metrics 仍可访问。
        base_chamfer = chamfer   # d_x2y.mean() + d_y2x.mean(), 无权重对照
        beta = 1.0

        # targeted transport: targeted_h_v3 需在 chamfer/L2 组装前拿到 source mask，
        # 以便对 source 点的 x2y / L2 部分降权（解除拔河）。此处预算一次 transport，
        # 结果在后面 transport 组装点复用（_v3_transport_cache）。
        self._v3_src_mask = None
        self._v3_transport_cache = None
        if self.transport_loss_mode == "targeted_h_v3":
            _tr_loss, _tr_metrics, _src_mask = _targeted_transport_loss(
                denoised, pc_clean_flat, K=self.transport_k)
            self._v3_transport_cache = (_tr_loss, _tr_metrics)
            self._v3_src_mask = _src_mask   # (B,N) numpy 0/1 或 None

        # density-loss probe dcd_like / dcd_official probes: when either is "on" replace
        # chamfer. Mutually exclusive (asserted at __init__).
        # emd / uniformcd 分支的观测指标容器（分支未激活时保持空 dict，
        # metrics 组装点 ** 展开为零影响）。
        emd_metrics = {}
        uniformcd_metrics = {}
        if self.dcd_like_loss_mode == "on":
            from .losses.dcd_like import compute_dcd_like_loss
            dcd_like = compute_dcd_like_loss(
                denoised, pc_clean_flat,
                alpha=self.dcd_like_alpha,
                n_lambda=self.dcd_like_n_lambda,
            )
            out[LOSS_KEY_DCD_LIKE] = dcd_like
        elif self.dcd_official_loss_mode == "on":
            raise ValueError("dcd_official is excluded from this release")
        elif self.emd_loss_mode == "on":
            # EMD 替换 chamfer（配方 0.1×EMD.sum()，
            # 0.1 由 task yaml 权重施加）。emd_metrics 在下方 metrics 组装点消费。
            from .losses.emd_sum import compute_emd_loss
            emd_loss, emd_metrics = compute_emd_loss(
                denoised, pc_clean_flat,
                eps=self.emd_eps, iters=self.emd_iters,
            )
            out[LOSS_KEY_EMD] = emd_loss
        elif self.uniformcd_loss_mode == "on":
            # UniformCD 替换 chamfer（密度 detach、
            # 对应搜索改配、L2 锚保留）。
            from .losses.uniformcd import compute_uniformcd_loss
            uniformcd_loss, uniformcd_metrics = compute_uniformcd_loss(
                denoised, pc_clean_flat,
                k=self.uniformcd_k,
            )
            out[LOSS_KEY_UNIFORMCD] = uniformcd_loss
        else:
        # completeness weighting FCD completeness weighting.
            # base_chamfer 和 beta 已在上面初始化 (α=β=1).
            if self.fcd_chamfer_mode != "off":
                beta = _fcd_beta(self.fcd_chamfer_mode, epoch,
                                 self.fcd_total_epochs, self.fcd_exp_sigma)
                chamfer = d_x2y.mean() + beta * d_y2x.mean()
            # targeted transport: 对 source 点的 x2y 半边部分降权（y2x 不动），
            # 解除「chamfer 把 H source 拉回原最近 clean」的拔河。
            if (self.transport_loss_mode == "targeted_h_v3"
                    and self._v3_src_mask is not None):
                x2y_dw = _source_downweighted_mean(
                    d_x2y, self._v3_src_mask, self.transport_source_x2y_alpha)
                chamfer = x2y_dw + beta * d_y2x.mean()
            out[LOSS_KEY_CHAMFER] = chamfer

        if self.uniformcd_nudge_mode == "on":
            # UniformCD 也可作为 Chamfer 之上的附加项（未用于提交）
            # （r clip 由 uniformcd_clip_lo/hi 控制；与四 replace lane 互斥，
            # 互斥断言在 __init__）。uniformcd_metrics 在下方 metrics 组装点消费。
            # 可选 backend=jittor（GPU 加速，数学式一致）。
            if self.uniformcd_backend == "jittor":
                from .losses.uniformcd import compute_uniformcd_loss_gpu
                uniformcd_loss, uniformcd_metrics = compute_uniformcd_loss_gpu(
                    denoised, pc_clean_flat,
                    k=self.uniformcd_k,
                    clip_lo=self.uniformcd_clip_lo, clip_hi=self.uniformcd_clip_hi,
                )
            else:
                from .losses.uniformcd import compute_uniformcd_loss
                uniformcd_loss, uniformcd_metrics = compute_uniformcd_loss(
                    denoised, pc_clean_flat,
                    k=self.uniformcd_k,
                    clip_lo=self.uniformcd_clip_lo, clip_hi=self.uniformcd_clip_hi,
                )
            out[LOSS_KEY_UNIFORMCD] = uniformcd_loss

        # L2 分支.
        warmup_active = (
            self.l2_target_mode == "pred_nn_clean"
            and epoch < self.l2_target_warmup_epochs
        )
        l2_target_raw_value = None  # 日志用; off 下保持 None 以记录此 step 未算 L2
        l2_weight_metrics = {}
        if self.l2_target_mode == "off":
            # loss_dict 不含 l2 key; spec.py forward 遍历 loss_dict 而非 loss_config,
            # 自然跳过. 任务 yaml 的 loss block 不应声明 l2 以保持审计一致.
            pass
        elif self.l2_target_mode == "fixed" or warmup_active:
            # paired baseline L2 / 最近邻 L2 target warmup 期也走此路径.
            l2_per_point = ((denoised - pc_clean_flat) ** 2).sum(dim=-1)
            if (self.transport_loss_mode == "targeted_h_v3"
                    and self._v3_src_mask is not None):
                # targeted transport: 对 source 点 fixed-L2 部分降权，与 x2y 一致解除拔河。
                l2 = _source_downweighted_mean(
                    l2_per_point, self._v3_src_mask, self.transport_source_l2_alpha)
                l2_weight_metrics = {"l2_v3_source_alpha": self.transport_source_l2_alpha}
            else:
                l2, l2_weight_metrics = _weighted_fixed_l2(
                    l2_per_point,
                    mode=self.l2_weight_mode,
                    topk_frac=self.l2_weight_topk_frac,
                    downweight=self.l2_weight_downweight,
                )
            out[LOSS_KEY_L2] = l2
            l2_target_raw_value = float(l2.item())
        elif self.l2_target_mode == "pred_nn_clean":
            # 最近邻 L2 target post-warmup: l2 target = pc_clean[idx_x2y].
            # nn_y 已经在 _chamfer_l2_breakdown 内 gather 过 (pc_clean_flat[bi, idx_x2y]),
            # 等价于 pc_clean[nn_idx]. idx_x2y 是 int 索引, 天然 stop-grad;
            # pc_clean_flat 是 ground-truth 不参与训练, 也等价 stop-grad.
            l2 = ((denoised - nn_y) ** 2).sum(dim=-1).mean()
            out[LOSS_KEY_L2] = l2
            l2_target_raw_value = float(l2.item())
        else:
            raise AssertionError(
                f"unreachable l2_target_mode={self.l2_target_mode!r}; "
                "init-time assert should have caught this"
            )

        keep_metrics = {}
        if self.keep_loss_mode == "soft_weight":
            keep_logit = getattr(self.network, "_keep_logit", None)
            if keep_logit is None:
                raise RuntimeError(
                    "keep_loss_mode=soft_weight requires network._keep_logit; "
                    "check keep_head_mode and model.execute()"
                )
            w = jt.sigmoid(keep_logit)  # (B*P, M, 1)
            # weighted_y2x 只训练保留权重预测头；坐标在此分离梯度。
            denoised_keep = denoised.detach()
            if self.chamfer_impl == "pairwise_legacy_control":
                _, _, d_y2x_keep, _, _, idx_y2x_keep = _chamfer_l2_breakdown_pairwise(
                    denoised_keep, pc_clean_flat
                )
            else:
                _, _, d_y2x_keep, _, _, idx_y2x_keep = _chamfer_l2_breakdown(
                    denoised_keep, pc_clean_flat
                )
            B_keep, M_keep = idx_y2x_keep.shape
            bi_keep = jt.arange(B_keep).view(-1, 1)
            w_sq = w.squeeze(-1)
            w_resp = w_sq[bi_keep, idx_y2x_keep]  # (B*P, M_clean)
            weighted_y2x = (d_y2x_keep / (w_resp + 1e-8)).mean()
            out[LOSS_KEY_WEIGHTED_Y2X] = weighted_y2x

            w_mean = w.mean()
            w_var = ((w - w_mean) ** 2).mean()
            keep_var_penalty = jt.maximum(jt.zeros_like(w_var), float(self.keep_var_floor) - w_var)
            keep_reg = ((w_mean - float(self.keep_target_keep)) ** 2) + (
                float(self.keep_var_lambda) * keep_var_penalty
            )
            out[LOSS_KEY_KEEP_REG] = keep_reg

            with jt.no_grad():
                eps = 1e-8
                w_det = w.detach()
                w_mean_val = float(w_mean.item())
                w_var_val = float(w_var.item())
                w_std_val = float(jt.sqrt(w_var + eps).item())
                w_entropy = -(
                    w_det * jt.log(w_det + eps) + (1.0 - w_det) * jt.log(1.0 - w_det + eps)
                ).mean()

                radii = jt.sqrt((pc_noisy_flat.detach() ** 2).sum(dim=-1))  # (B*P, M)
                w_np = w_det.squeeze(-1).numpy()
                radii_np = radii.numpy()
                boundary_ratios = []
                for b_idx in range(B_keep):
                    r_patch = radii_np[b_idx]
                    w_patch = w_np[b_idx]
                    boundary_thr = np.quantile(r_patch, 0.9)
                    boundary_mask = r_patch >= boundary_thr
                    inner_mask = ~boundary_mask
                    if boundary_mask.sum() == 0 or inner_mask.sum() == 0:
                        continue
                    inner_mean = float(w_patch[inner_mask].mean())
                    if inner_mean <= 1e-8:
                        continue
                    boundary_mean = float(w_patch[boundary_mask].mean())
                    boundary_ratios.append(boundary_mean / inner_mean)
                keep_boundary_ratio = float(np.mean(boundary_ratios)) if boundary_ratios else 1.0

                keep_metrics = {
                    "weighted_y2x_raw": float(weighted_y2x.item()),
                    "keep_reg_raw": float(keep_reg.item()),
                    "w_mean": w_mean_val,
                    "w_var": w_var_val,
                    "w_std": w_std_val,
                    "w_entropy": float(w_entropy.item()),
                    "w_resp_mean": float(w_resp.mean().item()),
                    "keep_boundary_ratio": keep_boundary_ratio,
                }

        # candidate-select-clean: slot-variant selector clean-guided 辅助 loss（per-slot R-way CE）。
        # 机制：逐点 argmax 对 logit 不可微，故 selector 由此 cross-entropy 监督。
        # target = 每个点 i 的 R 个变体中离 clean 最近的那个变体（逐点 R-way 分类）。
        # 这天然保槽位（只在点 i 自己的变体内分类），且 = "选离表面最近的变体"的可训练化。
        # clean 仅训练期构造 target；推理期 argmax，无 clean 泄漏。
        # 梯度链：CE -> MLP_selector -> predict_z(未 detach) -> 主干（避开 soft-weight detach 坑）。
        self._selector_metrics = {}
        variant_logit = getattr(self.network, "_variant_logit", None)
        if self.selector_aux_mode == "variant_ce" and variant_logit is not None:
            # variant_logit: (B, M, R)。需要重建变体坐标算每点最近变体 target。
            sel_var_idx = self.network._selected_variant_idx   # (B, M) 推理选择（诊断用）
            Bc, Mc, R = variant_logit.shape
            # 用 model stash 的变体张量算每点最近变体 target（paired_idx：clean 第 i 点对齐点 i）
            variants = getattr(self.network, "_slot_variants", None)
            if variants is not None:
                vd = variants.detach()                         # (B, M, R, 3)
                # 每点 R 变体到对应 clean 点的距离 —— 但 paired_idx 下 clean 第 i 点对齐点 i
                cl = pc_clean_flat.detach()                    # (B, M, 3)
                cl_e = cl.unsqueeze(2).broadcast([Bc, Mc, R, 3])
                d = ((vd - cl_e) ** 2).sum(dim=-1)             # (B, M, R) 各变体到 clean_i 的距离
                tgt, _ = jt.argmin(d, dim=2)                   # (B, M) 最近变体 = 正确类
                tgt = tgt.detach()
                # softmax CE：-log softmax(logit)[tgt]
                logp = jt.nn.log_softmax(variant_logit, dim=2)  # (B, M, R)
                bi = jt.arange(Bc).view(Bc, 1).broadcast([Bc, Mc])
                mi = jt.arange(Mc).view(1, Mc).broadcast([Bc, Mc])
                ce = (-logp[bi, mi, tgt]).mean()
                out[LOSS_KEY_SELECTOR_AUX] = ce
                with jt.no_grad():
                    # 推理选择命中 clean 最近变体的比例（selector 是否选对）
                    hit = float((sel_var_idx == tgt).float().mean().item())
                    # 非 base 变体被选中的比例（selector 是否真在用变体而非恒退 base）
                    nonbase = float((sel_var_idx != 0).float().mean().item())
                    self._selector_metrics = {
                        "selector_aux_raw": float(ce.item()),
                        "variant_R": int(R),
                        "selector_hit_clean_frac": hit,
                        "selector_nonbase_frac": nonbase,
                    }

        # direction loss + observation metrics (stash for system-layer logging)
        if self.direction_loss_variant != "off":
            pred_disp = denoised - pc_noisy_flat
            gt_disp = pc_clean_flat - pc_noisy_flat
            dir_out = _masked_direction_loss(
                pred_disp, gt_disp,
                tau_ratio=self.direction_mask_tau_ratio,
                variant=self.direction_loss_variant,
            )
            out[LOSS_KEY_DIR_COS] = dir_out["L_dir"]

            # 训练期近似 paired_L2_ratio, 与 gate (diag_paired_cd.py) 的
            # paired_L2_ratio 同口径: mean||denoised-clean|| / mean||noisy-clean||.
            # 区别于 disp_scale_masked (mean of per-point ratios over mask).
            resid_norm = ((denoised - pc_clean_flat) ** 2).sum(dim=-1).sqrt().mean()
            gt_disp_norm = ((pc_clean_flat - pc_noisy_flat) ** 2).sum(dim=-1).sqrt().mean()
            paired_l2_ratio_train = resid_norm / (gt_disp_norm + 1e-8)

            dir_metrics = {
                "L_dir_raw":             float(dir_out["L_dir"].item()),
                "L_mag_logonly":         float(dir_out["L_mag_logonly"].item()),
                "disp_cos_masked":       float(dir_out["disp_cos_masked"].item()),
                "disp_scale_masked":     float(dir_out["disp_scale_masked"].item()),
                "dir_mask_ratio":        float(dir_out["dir_mask_ratio"].item()),
                "paired_L2_ratio_train": float(paired_l2_ratio_train.item()),
            }
        else:
            dir_metrics = {}

        # L2 target rescue 日志六字段.
        # 所有 mode 一致写入, off 下 l2_target_raw=None; 保证 epoch_summary.jsonl
        # 跨 run schema 一致.
        if self.train_r3_metrics:
            with jt.no_grad():
            # per-point 距离 |denoised_i - clean[nn_idx_i]|
                match_dist_per_point = (
                    ((denoised - nn_y) ** 2).sum(dim=-1).sqrt()
                )
                match_dist_per_patch = match_dist_per_point.mean(dim=-1).numpy()  # (B*P,)
                nn_idx_np = idx_x2y_BN.numpy().astype(np.int64)                   # (B*P, N)
            B_total, N_pred = nn_idx_np.shape
            M_clean = int(pc_clean_flat.shape[1])
        # per-patch unique clean count / M_clean; 注意 §7.3 分母是 M_clean 而非 N_pred
        # (当前 N_pred == M_clean == 1024 二者相等, 但口径应与显式 pairwise 版本
        # 保持一致, 避免 patch_size 变化时潜伏 bug).
            covs = np.array(
                [np.unique(nn_idx_np[b]).size / float(M_clean) for b in range(B_total)],
                dtype=np.float64,
            )
            coverage_mean = float(covs.mean())
            duplicate_mean = float((1.0 - covs).mean())
            mean_match_dist = float(match_dist_per_patch.astype(np.float64).mean())

            r3_metrics = {
                "chamfer_x2y_raw":       float(d_x2y.mean().item()),
                "chamfer_y2x_raw":       float(d_y2x.mean().item()),
                "l2_target_raw":         l2_target_raw_value,
                "l2_target_active":      (0.0 if self.l2_target_mode == "off" else 1.0),
                "coverage_ratio_train":  coverage_mean,
                "duplicate_ratio_train": duplicate_mean,
                "mean_match_dist_train": mean_match_dist,
            }
        else:
            r3_metrics = {}
        # 注: `l2_target_raw` 在 off 下为 None, outer system 的 float(val) 会 TypeError 跳过,
        # 因此 `epoch_summary.jsonl` 里 off run 不会出现 `l2_target_raw_mean` (设计如此,
        # 明确表示 "该 step 未计算 l2 target"). `l2_target_active` 作为数值 flag 保证 off run
        # 仍能从 jsonl 看出 L2 是否参与, schema 对审计仍完整.

        # coverage-hole loss .
        # 仅 mode != "off" 时进入 loss_dict; mode==off 保持空 cov_metrics.
        if self.coverage_loss_mode == "hole_top10":
            M_y2x = d_y2x.shape[-1]
            k = max(1, int(M_y2x * self.coverage_top_k_frac))
            L_cov = _topk_mean_lastdim(d_y2x, k)
            out[LOSS_KEY_COVERAGE] = L_cov

            # Tail ratio: top-k mean / overall mean. 训练初期应 5+ (long-tail holes 存在);
            # 训练中后期若降到 ~1, 说明 coverage-hole 已把 long-tail 拉平 (success signal);
            # 若反向升到 10+, 极端 hole 反复惩罚, 可能 over-fitting.
            mean_d_y2x_val = float(d_y2x.mean().item())
            L_cov_val = float(L_cov.item())
            tail_ratio = L_cov_val / max(mean_d_y2x_val, 1e-20)
            cov_metrics = {
                "L_coverage_raw":            L_cov_val,
                "coverage_hole_topk_raw":    L_cov_val,
                "coverage_mean_y2x_raw":     mean_d_y2x_val,
                "coverage_tail_ratio_train": tail_ratio,
                "coverage_top_k_frac":       float(self.coverage_top_k_frac),
            }
        elif self.coverage_loss_mode == "off":
            cov_metrics = {}
        else:
            raise AssertionError(
                f"unreachable coverage_loss_mode={self.coverage_loss_mode!r}; "
                "init-time assert should have caught this"
            )

        # Sliced Wasserstein loss (third term only).
        if self.sw_loss_variant != "off":
            K = self.sw_num_projections
            # --- projection schedule ---
            if self.sw_projection_schedule == "per_epoch":
                if self._cached_sw_theta is None:
                    if self.sw_projection_scope == "per_sample":
                        B = denoised.shape[0]
                        self._cached_sw_theta = _sample_per_sample_projections(B, 3, K)
                    else:
                        self._cached_sw_theta = _sample_sw_projections(3, K)
                theta = self._cached_sw_theta
            else:
                theta = None  # per_step: sample inside loss function

            # --- compute ---
            if self.sw_loss_variant == "vanilla_global_shared":
                sw_loss = _sw_loss_vanilla(denoised, pc_clean_flat, theta=theta, K=K)
            elif self.sw_loss_variant == "vanilla_per_sample":
                sw_loss = _sw_loss_per_sample(denoised, pc_clean_flat, theta=theta, K=K)
            else:
                raise AssertionError(f"unreachable sw_loss_variant={self.sw_loss_variant!r}")
            out[LOSS_KEY_SW] = sw_loss
            sw_metrics = {"L_sw_raw": float(sw_loss.item())}
        else:
            sw_metrics = {}

        # PU-Net repulsion loss (third term only, dry-run calibrated).
        if self.repulsion_loss_variant == "punet":
            rep_loss = _repulsion_loss_train(
                denoised, k=self.repulsion_k,
                radius=self.repulsion_radius, h=self.repulsion_h,
            )
            out[LOSS_KEY_REPULSION] = rep_loss
            rep_metrics = {"L_rep_raw": float(rep_loss.item())}
        elif self.repulsion_loss_variant == "off":
            rep_metrics = {}
        else:
            raise AssertionError(
                f"unreachable repulsion_loss_variant={self.repulsion_loss_variant!r}; "
                "init-time assert should have caught this"
            )

        # centroid-anchor: centroid anchor loss (替代 fixed_L2 的弱稳定项).
        # 仅 target_mode != paired_idx 且 centroid_anchor_weight > 0 时启用.
        if self.target_mode != "paired_idx" and self.centroid_anchor_weight > 0:
            centroid_pred = denoised.mean(dim=1)                    # (B*P, 3)
            centroid_target = pc_clean_flat.mean(dim=1)              # (B*P, 3)
            L_centroid = ((centroid_pred - centroid_target) ** 2).sum(dim=-1).mean()
            out[LOSS_KEY_CENTROID_ANCHOR] = L_centroid
            centroid_metrics = {
                "L_centroid_anchor_raw": float(L_centroid.item()),
                "centroid_anchor_weight": self.centroid_anchor_weight,
            }
        else:
            centroid_metrics = {}

        # Hungarian one-to-one assignment loss (third term only, 2026-05-20).
        if self.hungarian_loss_variant == "vanilla":
            if self.hungarian_n_eval < denoised.shape[1]:
                # stride subset (same indices per sample, deterministic)
                n_eval = self.hungarian_n_eval
                N_full = denoised.shape[1]
                idx_np = np.linspace(0, N_full - 1, n_eval).astype(np.int64)
                idx_jt = jt.array(idx_np)
                denoised_c5 = denoised[:, idx_jt, :]
                clean_c5 = pc_clean_flat[:, idx_jt, :]
            else:
                denoised_c5 = denoised
                clean_c5 = pc_clean_flat
            c5_loss, _c5_col_np = _hungarian_matched_l2_loss(denoised_c5, clean_c5)
            out[LOSS_KEY_C5_HUNGARIAN] = c5_loss
            c5_metrics = {"L_c5_hungarian_raw": float(c5_loss.item())}
        elif self.hungarian_loss_variant == "local_subpatch":
            if self.hungarian_n_eval < denoised.shape[1]:
                raise ValueError(
                    f"local_subpatch forbids hungarian_n_eval ({self.hungarian_n_eval}) "
                    f"< patch_size ({denoised.shape[1]}); stride subset breaks local groups"
                )
            if denoised.shape[1] % self.hungarian_subpatch_size != 0:
                raise ValueError(
                    f"patch_size ({denoised.shape[1]}) must be divisible by "
                    f"hungarian_subpatch_size ({self.hungarian_subpatch_size})"
                )
            c5_loss, _c5_col_np = _hungarian_subpatch_matched_l2_loss(
                denoised, pc_clean_flat,
                group_size=self.hungarian_subpatch_size,
            )
            out[LOSS_KEY_C5_HUNGARIAN] = c5_loss
            c5_metrics = {"L_c5_hungarian_raw": float(c5_loss.item())}
        elif self.hungarian_loss_variant == "off":
            c5_metrics = {}
        else:
            raise AssertionError(
                f"unreachable hungarian_loss_variant={self.hungarian_loss_variant!r}; "
                "init-time assert should have caught this"
            )

        # targeted transport (third term only). 独立于通用 Hungarian lane。
        if self.transport_loss_mode == "targeted_h":
            tr_loss, tr_metrics_raw, _src_mask = _targeted_transport_loss(
                denoised, pc_clean_flat, K=self.transport_k)
            if tr_loss is not None:
                out[LOSS_KEY_TRANSPORT_H] = tr_loss
                transport_metrics = {"L_transport_h_raw": float(tr_loss.item()),
                                     **tr_metrics_raw}
            else:
                transport_metrics = {"L_transport_h_raw": 0.0, **tr_metrics_raw}
        elif self.transport_loss_mode == "targeted_h_v3":
            # 复用前面（chamfer 组装前）预算的 transport，保证 source 选择与 x2y/L2
            # 降权用的是同一批点（同一次 H forward）。
            tr_loss, tr_metrics_raw = self._v3_transport_cache
            if tr_loss is not None:
                out[LOSS_KEY_TRANSPORT_H] = tr_loss
                transport_metrics = {"L_transport_h_raw": float(tr_loss.item()),
                                     **tr_metrics_raw}
            else:
                transport_metrics = {"L_transport_h_raw": 0.0, **tr_metrics_raw}
        else:
            transport_metrics = {}

        # dense-surface auxiliary dense surface auxiliary (third term only, stacked on paired base).
        if self.dense_aux_loss_mode != "off":
            if "pc_clean_dense_local" not in batch:
                raise RuntimeError(
                    f"dense_aux_loss_mode={self.dense_aux_loss_mode!r} but batch has no "
                    f"'pc_clean_dense_local'; check transform AugmentPatch.dense_target_size>0 "
                    f"and AugmentSample.dense_clean_target_samples>0"
                )
            # (B, P, Md, 3) -> (B*P, Md, 3); Md 由 transform dense_target_size 决定。
            dense_local = batch["pc_clean_dense_local"]
            dense_md = dense_local.shape[-2]
            dense_local_flat = dense_local.reshape(-1, dense_md, 3)
            dense_loss, dense_aux_metrics = _dense_surface_aux_loss(
                denoised, dense_local_flat, variant=self.dense_aux_loss_mode
            )
            out[LOSS_KEY_DENSE_AUX] = dense_loss
            dense_aux_metrics = {
                "L_dense_aux_raw": float(dense_loss.item()),
                "dense_aux_md": float(dense_md),
                **dense_aux_metrics,
            }
        else:
            dense_aux_metrics = {}

        # log-only direction diagnostics: 只在 direction_head_mode=="replace_output" 时记录。
        # 不上梯度、不进 loss_dict。
        dh1_metrics = {}
        _head_mode = getattr(self.network, "direction_head_mode", "off")
        if _head_mode in ("replace_output", "auxiliary"):
            with jt.no_grad():
                gt_disp = pc_clean_flat - pc_noisy_flat
                gt_norm = (gt_disp ** 2).sum(dim=-1).sqrt()
                _eps = 1e-8

                # head displacement: from stash (auxiliary) or from denoised (replace_output)
                if _head_mode == "auxiliary":
                    head_mag = self.network._dh1_head_mag
                    head_unit_dir = self.network._dh1_head_unit_dir
                    if head_mag is not None and head_unit_dir is not None:
                        pred_disp_head = head_mag * head_unit_dir
                        head_denoised = pc_noisy_flat + pred_disp_head
                    else:
                        pred_disp_head = denoised - pc_noisy_flat
                        head_denoised = denoised
                else:
                    pred_disp_head = denoised - pc_noisy_flat
                    head_denoised = denoised

                pred_norm_head = (pred_disp_head ** 2).sum(dim=-1).sqrt()
                cos_head = (pred_disp_head * gt_disp).sum(dim=-1) / (pred_norm_head * gt_norm + _eps)
                dh1_metrics["dh1_head_disp_cos_train"] = float(cos_head.mean().item())
                dh1_metrics["dh1_head_disp_scale_train"] = float(
                    (pred_norm_head / (gt_norm + _eps)).mean().item()
                )

                denoised_implicit = self.network._dh1_denoised_implicit
                if denoised_implicit is not None:
                    pred_disp_imp = denoised_implicit - pc_noisy_flat
                    pred_norm_imp = (pred_disp_imp ** 2).sum(dim=-1).sqrt()
                    cos_imp = (pred_disp_imp * gt_disp).sum(dim=-1) / (pred_norm_imp * gt_norm + _eps)
                    dh1_metrics["dh1_implicit_disp_cos_train"] = float(cos_imp.mean().item())
                    dh1_metrics["dh1_implicit_disp_scale_train"] = float(
                        (pred_norm_imp / (gt_norm + _eps)).mean().item()
                    )
                    delta_l2 = ((head_denoised - denoised_implicit) ** 2).sum(dim=-1).sqrt().mean()
                    dh1_metrics["dh1_head_implicit_delta_l2"] = float(delta_l2.item())

        # direction-magnitude supervised losses: head-level unit_dir + mag supervision.
        if self.dh1_dir_loss_mode != "off" or self.dh1_mag_loss_mode != "off":
            pred_disp_head = denoised - pc_noisy_flat
            gt_disp = pc_clean_flat - pc_noisy_flat
            if self.dh1_dir_target_mode == "nn_clean":
                with jt.no_grad():
                    _, nn_idx = safe_knn(pc_noisy_flat, pc_clean_flat, 1)
                    B_val, M_val = pc_noisy_flat.shape[0], pc_noisy_flat.shape[1]
                    bi = jt.arange(B_val).unsqueeze(-1).broadcast([B_val, M_val])
                    nn_clean = pc_clean_flat[bi, nn_idx.reshape(B_val, M_val)]
                gt_disp = nn_clean - pc_noisy_flat
            if self.dh1_dir_loss_mode == "masked_cos":
                pred_unit_dir = self.network._dh1_head_unit_dir
                if pred_unit_dir is None:
                    pred_unit_dir = pred_disp_head / (
                        (pred_disp_head ** 2).sum(dim=-1, keepdims=True).sqrt() + 1e-4
                    )
                dir_out = _dh1_unit_dir_loss(pred_unit_dir, gt_disp)
                out[LOSS_KEY_DH1_DIR] = dir_out["L_dir"]
                dh1_metrics["dh1_dir_L_raw"] = float(dir_out["L_dir"].item())
                dh1_metrics["dh1_dir_cos_masked"] = float(dir_out["disp_cos_masked"].item())
                dh1_metrics["dh1_dir_mask_ratio"] = float(dir_out["dir_mask_ratio"].item())
            if self.dh1_mag_loss_mode != "off":
                if self.dh1_mag_loss_mode == "stable_huber_log":
                    pred_mag = self.network._dh1_head_mag
                    if pred_mag is None:
                        raise RuntimeError(
                            "dh1_mag_loss_mode=stable_huber_log requires "
                            "direction_head_mode=replace_output and network._dh1_head_mag"
                        )
                    mag_out = _dh1_stable_mag_loss(pred_mag, gt_disp)
                else:
                    mag_out = _dh1_mag_loss(
                        pred_disp_head, gt_disp,
                        variant=self.dh1_mag_loss_mode,
                    )
                out[LOSS_KEY_DH1_MAG] = mag_out["L_mag"]
                dh1_metrics["dh1_mag_L_raw"] = float(mag_out["L_mag"].item())
                dh1_metrics["dh1_mag_ratio"] = float(mag_out["mag_ratio"].item())

        # post-FBM residual head diagnostics
        if getattr(self.network, "residual_head_mode", "off") == "post_fbm":
            residual_metrics = {
                "post_delta_norm_mean": float(self.network._residual_delta_norm or 0.0),
                "post_delta_scale_vs_implicit_disp": float(self.network._residual_delta_ratio or 0.0),
            }
        else:
            residual_metrics = {}

        pull_metrics = {}
        if getattr(self.network, "pull_head_mode", "off") == "confidence_delta":
            pull_w = getattr(self.network, "_pull_w", None)
            pull_norm = getattr(self.network, "_pull_norm", None)
            if pull_w is not None and pull_norm is not None:
                pull_metrics = {
                    "pull_w_mean": float(pull_w.mean().item()),
                    "pull_norm_mean": float(pull_norm.mean().item()),
                    "pull_input_allowed_only": bool(
                        getattr(self.network, "_pull_input_audit", {}).get("allowed_inputs_only", False)
                    ),
                }

        # FCD 诊断: 始终记录半边 Chamfer / beta / base_chamfer，无论 mode.
        fcd_metrics = {
            "L_chamfer_base_raw": float(base_chamfer.item()),
            "chamfer_x2y_raw":    float(d_x2y.mean().item()),
            "chamfer_y2x_raw":    float(d_y2x.mean().item()),
            "fcd_beta":           float(beta),
        }
        if self.fcd_chamfer_mode != "off":
            fcd_metrics["fcd_y2x_weighted_raw"] = float((beta * d_y2x.mean()).item())
        if self.fcd_chamfer_mode == "stair":
            half_ep = float(self.fcd_total_epochs) / 2.0
            fcd_metrics["fcd_stair_threshold"] = half_ep

        # 合并 base、方向、coverage、soft-weight、repulsion 和密度诊断指标。
        self._last_train_metrics = {
            "L_chamfer_raw": float(chamfer.item()),
            "L_l2_raw":      (l2_target_raw_value if l2_target_raw_value is not None else 0.0),
            **fcd_metrics,
            **l2_weight_metrics,
            **r3_metrics,
            **dir_metrics,
            **cov_metrics,
            **sw_metrics,
            **rep_metrics,
            **centroid_metrics,
            **c5_metrics,
            **transport_metrics,
            **dense_aux_metrics,
            **keep_metrics,
            **self._selector_metrics,
            **dh1_metrics,
            **residual_metrics,
            **pull_metrics,
        }
        if self.dcd_like_loss_mode == "on":
            self._last_train_metrics["L_dcd_like_raw"] = float(dcd_like.item())
        elif self.dcd_official_loss_mode == "on":
            self._last_train_metrics["L_dcd_official_raw"] = float(dcd_official.item())
        elif self.emd_loss_mode == "on":
            # EMD 分支观测项：assignment 唯一率
            self._last_train_metrics["L_emd_raw"] = float(emd_loss.item())
            self._last_train_metrics.update(emd_metrics)
        elif self.uniformcd_loss_mode == "on":
            # UniformCD 分支观测项：对应变更率 / 重复代理 / r 标准差
            self._last_train_metrics["L_uniformcd_raw"] = float(uniformcd_loss.item())
            self._last_train_metrics.update(uniformcd_metrics)
        if self.uniformcd_nudge_mode == "on":
            # UniformCD 附加项观测项（同上 + r 裁剪命中率）
            self._last_train_metrics["L_uniformcd_raw"] = float(uniformcd_loss.item())
            self._last_train_metrics.update(uniformcd_metrics)

        # 全局上下文统计
        if self.network.global_context_mode != "off":
            assert self.network._global_diag, (
                "global_context_mode enabled but _global_diag empty; "
                "check model.execute() order"
            )
            self._last_train_metrics.update({
                "global_gamma": self.network._global_diag["global_gamma"],
                "global_delta_norm": self.network._global_diag["global_delta_norm"],
                "global_feature_mlgc_norm": self.network._global_diag["feature_mlgc_norm"],
                "global_delta_ratio": self.network._global_diag["global_delta_ratio"],
                "global_applied_delta_ratio": self.network._global_diag["global_applied_delta_ratio"],
                "global_attn_entropy_mean": self.network._global_diag["attention_entropy_mean"],
            })

        # GroupToken 骨干网络统计
        if getattr(self.network, "backbone_mode", "mlgc") == "group_token_hilbert":
            group_diag = self.network.group_token_backbone.last_diag()
            self._last_train_metrics.update({
                "group_token_C": group_diag.get("token_C", 0.0),
                "group_token_G": group_diag.get("token_G", 0.0),
                "group_token_2G": group_diag.get("token_2G", 0.0),
                "group_token_aug_norm": group_diag.get("aug_delta_norm", 0.0),
                "group_token_mixer_entropy": group_diag.get("mixer_entropy_mean", 0.0),
                "group_token_upsample_entropy": group_diag.get("upsample_entropy_mean", 0.0),
                "group_token_usage_ratio": group_diag.get("upsample_token_usage_ratio", 0.0),
                "group_token_center_usage_ratio": group_diag.get("upsample_center_usage_ratio", 0.0),
                "group_token_point_id_contrib": group_diag.get("point_id_contrib_ratio", 0.0),
                "group_token_point_id_gamma": group_diag.get("point_id_gamma", 0.0),
                "group_token_point_id_norm": group_diag.get("point_id_norm", 0.0),
            })

        return out

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    # ---------- 推理 ----------
    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        """推理单个 noisy 点云 -> denoised (N, 3)。

        走 denoise.py 的推理流程:
            normalize_unit_sphere -> patch_denoise -> denormalize_unit_sphere
        严格保证 denoised.shape == pc_noisy.shape (赛题硬约束).

详见 denoise.py 顶部的推理侧硬约束.
        """
        from .denoise import denoise_full_cloud

        pc_noisy_batch = batch["pc_noisy"]               # (B, N, 3)
        patch_size = int(self.model_config.get("patch_size", 1024))
        seed_k = int(self.model_config.get("predict_seed_k", 3))
        seed_k_alpha = int(self.model_config.get("predict_seed_k_alpha", 5))
        return_pull_sidecars = bool(self.model_config.get(
            "predict_return_pull_sidecars",
            getattr(self.network, "pull_head_mode", "off") != "off",
        ))

        res = []
        for i in range(pc_noisy_batch.shape[0]):
            pc_noisy = pc_noisy_batch[i]                 # (N, 3)
            pc_noisy_np = pc_noisy.numpy().astype(np.float32)
            denoise_out = denoise_full_cloud(
                self.network,
                pc_noisy_np,
                patch_size=patch_size,
                seed_k=seed_k,
                seed_k_alpha=seed_k_alpha,
                return_coverage=True,
                return_sidecars=return_pull_sidecars,
            )
            if return_pull_sidecars:
                denoised_np, coverage_info, sidecars = denoise_out
            else:
                denoised_np, coverage_info = denoise_out
                sidecars = None
            # 硬不变量: shape 必须不变
            assert denoised_np.shape == pc_noisy_np.shape, (
                f"denoise_full_cloud changed shape! in={pc_noisy_np.shape}, "
                f"out={denoised_np.shape}"
            )
            rec = {
                "pc_denoised": denoised_np,
                "coverage_info": coverage_info,
            }
            if sidecars is not None:
                rec.update(sidecars)
                rec["pull_input_audit"] = getattr(self.network, "_pull_input_audit", {})
            res.append(rec)
        return res
