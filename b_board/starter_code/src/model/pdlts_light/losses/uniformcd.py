"""UniformCD（密度比对应搜索）对照损失，未用于最终提交。

依据 ECCV 2024 论文 Towards a Density Preserving Objective Function
for Learning on Point Sets 的式 (4)：

    dens(p,S) = 1 / max(sum(||p - kNN_S(p)||^2), dens_eps)
    r_y = dens(y,X) / dens(y,Y)
    r_x = dens(x,Y) / dens(x,X)

密度比参与最近邻选择，再对选中点对计算双向平方距离均值。
密度估计与离散匹配由 NumPy 计算，不参与求导；选中点对的坐标差
由 Jittor 计算并保留梯度。force_unit_ratio=True 将密度比设为 1，
用于与普通双向 Chamfer 比较。
"""

from __future__ import annotations

import numpy as np
import jittor as jt


def _np_pairwise_d2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(B,M,3),(B,N,3) -> (B,M,N) 平方距离矩阵（非负截断）。"""
    aa = (a ** 2).sum(-1, keepdims=True)
    bb = (b ** 2).sum(-1)[:, None, :]
    d2 = aa + bb - 2.0 * np.einsum("bmc,bnc->bmn", a, b)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _np_knn_sum_sq(d2: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    """每行到列侧 k 近邻的平方距离和（密度分母）。exclude_self 假定方阵同序。"""
    if exclude_self:
        d2 = d2.copy()
        idx = np.arange(d2.shape[1])
        d2[:, idx, idx] = np.inf
    part = np.argpartition(d2, k - 1, axis=2)[:, :, :k]
    return np.take_along_axis(d2, part, axis=2).sum(-1)


def _jt_knn_sum_sq(d2: "jt.Var", k: int, exclude_self: bool) -> "jt.Var":
    """GPU 版 _np_knn_sum_sq：argsort 全排后取前 k（M=2048 时 GPU 上足够快）。

    数学语义与 numpy 版（argpartition）一致：k 近邻平方距离和；
     tie 顺序差异不影响求和结果。
    """
    if exclude_self:
        m = d2.shape[1]
        eye = jt.array(np.eye(m, dtype=np.float32)[None] * 1e30)
        d2 = d2 + eye
    idx = jt.argsort(d2, dim=2, descending=False)[0][:, :, :k]
    topk_vals = jt.gather(d2, 2, idx)
    return topk_vals.sum(-1)


def compute_uniformcd_loss_gpu(pred: jt.Var, clean: jt.Var, k: int = 8,
                               dens_eps: float = 1e-30,
                               clip_lo: float | None = None,
                               clip_hi: float | None = None):
    """GPU 版 UniformCD（2026-07-24 验证）。

    数学语义与 compute_uniformcd_loss 逐式一致（同 docstring 公式），
    仅计算位置从 CPU numpy(f64) 挪到 GPU jittor(f32)：
      - 密度/对应搜索的输入经 .numpy() 取值（等价 stop_grad，密度不传梯度），
        同步量仅 (B,M,3) 小数组；(B,M,M) 距离矩阵/kNN/argmin 全部留在 GPU；
      - gather 坐标差走计算图（梯度通路）与 numpy 版完全一致；
      - metrics 与 numpy 版同口径（corr_chg/r_std/dup_proxy/r_clip_hit）。
    f32 vs f64 的 r 微小差异可能翻转个别 argmin 对应（loss 偏差 <<1%，
    数值对比 smoke 见 p190_u3_gpu_backend_smoke.py），判读口径不受影响。
    """
    x_np = pred.numpy().astype(np.float32)
    y_np = clean.numpy().astype(np.float32)
    B, M, _ = x_np.shape
    x = jt.array(x_np)                                  # detach 语义（密度用）
    y = jt.array(y_np)

    aa = (x ** 2).sum(-1, keepdims=True)                # (B,M,1)
    bb = (y ** 2).sum(-1).unsqueeze(1)                  # (B,1,M)
    d2_xy = aa + bb - 2.0 * jt.matmul(x, y.permute(0, 2, 1))
    d2_xy = jt.maximum(d2_xy, 0.0)                      # (B,M,M) 行=x 列=y

    dens_y_X = 1.0 / jt.maximum(_jt_knn_sum_sq(d2_xy, k, False), dens_eps)
    dens_x_Y = 1.0 / jt.maximum(
        _jt_knn_sum_sq(d2_xy.permute(0, 2, 1), k, False), dens_eps)
    aa_x = (x ** 2).sum(-1, keepdims=True)
    d2_xx = aa_x + aa_x.permute(0, 2, 1) - 2.0 * jt.matmul(x, x.permute(0, 2, 1))
    d2_xx = jt.maximum(d2_xx, 0.0)
    aa_y = (y ** 2).sum(-1, keepdims=True)
    d2_yy = aa_y + aa_y.permute(0, 2, 1) - 2.0 * jt.matmul(y, y.permute(0, 2, 1))
    d2_yy = jt.maximum(d2_yy, 0.0)
    dens_x_X = 1.0 / jt.maximum(_jt_knn_sum_sq(d2_xx, k, True), dens_eps)
    dens_y_Y = 1.0 / jt.maximum(_jt_knn_sum_sq(d2_yy, k, True), dens_eps)
    r_y = dens_y_X / dens_y_Y                           # (B,M) candidate=y 侧
    r_x = dens_x_Y / dens_x_X                           # (B,M) candidate=x 侧

    r_clip_hit = 0.0
    if clip_lo is not None or clip_hi is not None:
        lo = -float("inf") if clip_lo is None else float(clip_lo)
        hi = float("inf") if clip_hi is None else float(clip_hi)
        r_np = np.concatenate([r_y.numpy().ravel(), r_x.numpy().ravel()])
        r_clip_hit = float(((r_np < lo) | (r_np > hi)).mean())
        r_y = jt.clamp(r_y, lo, hi)
        r_x = jt.clamp(r_x, lo, hi)

    D = jt.sqrt(d2_xy)
    plain_xy = jt.argmin(D, dim=2)[0]                   # (B,M) 普通 Chamfer 对应
    plain_yx = jt.argmin(D.permute(0, 2, 1), dim=2)[0]
    uni_xy = jt.argmin(D * r_y.unsqueeze(1), dim=2)[0]  # (B,M)
    uni_yx = jt.argmin(D.permute(0, 2, 1) * r_x.unsqueeze(1), dim=2)[0]

    # Jittor 侧：gather 坐标差（梯度通路），与 numpy 版同模式
    bi = jt.arange(B).view(-1, 1)
    clean_sel = clean[bi, uni_xy]                       # (B,M,3)
    pred_sel = pred[bi, uni_yx]                         # (B,M,3)
    loss_x2y = ((pred - clean_sel) ** 2).sum(dim=-1).mean()
    loss_y2x = ((pred_sel - clean) ** 2).sum(dim=-1).mean()

    # metrics（小数组同步，与 numpy 版同口径）
    plain_xy_np = plain_xy.numpy()
    uni_xy_np = uni_xy.numpy()
    uni_yx_np = uni_yx.numpy()
    plain_yx_np = plain_yx.numpy()
    dup = [1.0 - np.unique(plain_xy_np[b]).size / M for b in range(B)]
    metrics = {
        "uniformcd_corr_chg_xy": float((uni_xy_np != plain_xy_np).mean()),
        "uniformcd_corr_chg_yx": float((uni_yx_np != plain_yx_np).mean()),
        "uniformcd_r_std": float(np.concatenate(
            [r_y.numpy().ravel(), r_x.numpy().ravel()]).std()),
        "uniformcd_dup_proxy": float(np.mean(dup)),
        "uniformcd_r_clip_hit": r_clip_hit,
    }
    return loss_x2y + loss_y2x, metrics


def compute_uniformcd_loss(pred: jt.Var, clean: jt.Var, k: int = 8,
                           dens_eps: float = 1e-30,
                           force_unit_ratio: bool = False,
                           clip_lo: float | None = None,
                           clip_hi: float | None = None):
    """UniformCD loss + 机制观测指标。

    Args:
        pred:  (B, M, 3) 去噪 patch（有梯度）
        clean: (B, M, 3) GT patch（无梯度），与 pred 同点数
        k:     kNN 密度邻居数（默认 8）
        dens_eps: 密度分母下限（探针同值 1e-30）
        force_unit_ratio: 数值等价 smoke 开关，r 强制为 1（应 == 官方 Chamfer）
        clip_lo/clip_hi: r 裁剪区间（            历史实验使用 clip [0.5,2.0]，依据前置探针
            膨胀率 4.97→1.71×、变更率仍 38.3%）。默认 None = 不裁剪，
            保持未裁剪的密度比。

    Returns:
        loss:    标量 jt.Var = loss_x2y + loss_y2x
        metrics: dict，含相对普通 argmin 的对应变更率、
                 r 的 std、dup proxy（普通 NN 下被多个 pred 共享的 clean 占比）、
                 r_clip_hit（被裁剪的 r 占比，未启用裁剪时恒 0）
    """
    # 注意：Jittor 的 stop_grad() 是原地生效的（会把 pred 从计算图摘出，
    # 2026-07-23 smoke 实测梯度被清零），这里直接 .numpy() 取值，不动图。
    x_np = pred.numpy().astype(np.float64)
    y_np = clean.numpy().astype(np.float64)
    B, M, _ = x_np.shape

    d2_xy = _np_pairwise_d2(x_np, y_np)                 # (B,M,M) 行=x 列=y

    if force_unit_ratio:
        r_y = np.ones((B, M), dtype=np.float64)
        r_x = np.ones((B, M), dtype=np.float64)
    else:
        dens_y_X = 1.0 / np.maximum(_np_knn_sum_sq(d2_xy, k, False), dens_eps)
        dens_x_Y = 1.0 / np.maximum(
            _np_knn_sum_sq(d2_xy.transpose(0, 2, 1), k, False), dens_eps)
        dens_x_X = 1.0 / np.maximum(
            _np_knn_sum_sq(_np_pairwise_d2(x_np, x_np), k, True), dens_eps)
        dens_y_Y = 1.0 / np.maximum(
            _np_knn_sum_sq(_np_pairwise_d2(y_np, y_np), k, True), dens_eps)
        r_y = dens_y_X / dens_y_Y                       # (B,M) candidate=y 侧
        r_x = dens_x_Y / dens_x_X                       # (B,M) candidate=x 侧

    # 可选密度传输项：r 裁剪（诊断窗口，默认不启用）。
    # 裁剪只作用于对应搜索的权重，不作用于 loss 本身（loss 仍是选中点对的距离）。
    r_clip_hit = 0.0
    if clip_lo is not None or clip_hi is not None:
        lo = -np.inf if clip_lo is None else float(clip_lo)
        hi = np.inf if clip_hi is None else float(clip_hi)
        r_all = np.concatenate([r_y.ravel(), r_x.ravel()])
        r_clip_hit = float(((r_all < lo) | (r_all > hi)).mean())
        r_y = np.clip(r_y, lo, hi)
        r_x = np.clip(r_x, lo, hi)

    D = np.sqrt(d2_xy)
    plain_xy = np.argmin(D, axis=2)                     # (B,M) 普通 Chamfer 对应
    plain_yx = np.argmin(D.transpose(0, 2, 1), axis=2)  # (B,M)
    uni_xy = np.argmin(D * r_y[:, None, :], axis=2)                 # (B,M)
    uni_yx = np.argmin(D.transpose(0, 2, 1) * r_x[:, None, :], axis=2)

    # Jittor 侧：gather 坐标差（梯度通路），与 chamfer 的 nn_y gather 同模式
    bi = jt.arange(B).view(-1, 1)
    clean_sel = clean[bi, jt.array(uni_xy.astype(np.int32))]        # (B,M,3)
    pred_sel = pred[bi, jt.array(uni_yx.astype(np.int32))]          # (B,M,3)
    loss_x2y = ((pred - clean_sel) ** 2).sum(dim=-1).mean()
    loss_y2x = ((pred_sel - clean) ** 2).sum(dim=-1).mean()

    # dup proxy：普通 NN 下，被 ≥2 个 pred 点选为最近的 clean 点占比
    dup = [1.0 - np.unique(plain_xy[b]).size / M for b in range(B)]
    metrics = {
        "uniformcd_corr_chg_xy": float((uni_xy != plain_xy).mean()),
        "uniformcd_corr_chg_yx": float((uni_yx != plain_yx).mean()),
        "uniformcd_r_std": float(np.concatenate([r_y.ravel(), r_x.ravel()]).std()),
        "uniformcd_dup_proxy": float(np.mean(dup)),
        "uniformcd_r_clip_hit": r_clip_hit,
    }
    return loss_x2y + loss_y2x, metrics
