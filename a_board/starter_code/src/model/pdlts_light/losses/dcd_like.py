"""早期密度加权 Chamfer 对照，公式不同于官方 DCD。

本实现使用 exp(-d / alpha) 的自归一化权重和线性混合系数 n_lambda，
损失为 d / w。官方 DCD 使用 exp(-d * alpha)、最近邻计数权重和有界损失。
两者的参数含义和梯度均不同，历史结果分别记录于
experiments/a_board_atlas.md。

该分支未用于最终提交。官方 DCD 的逐行改写未随仓库分发，
许可说明见根目录 THIRD_PARTY_NOTICES.md。
"""

from __future__ import annotations

import jittor as jt

from ..layer import safe_knn


def _one_side_dcd_like(
    src: jt.Var,
    tgt: jt.Var,
    alpha: float,
    n_lambda: float,
) -> jt.Var:
    """One-directional dcd_like (src -> tgt). NOT official DCD.

    Args:
        src: (B, N, 3)
        tgt: (B, M, 3)
        alpha: dcd_like temperature (note: enters as exp(-d/alpha), opposite
            to official calc_dcd which uses exp(-d*alpha)).
        n_lambda: dcd_like mixing coefficient in [0, 1] (linear mix; differs
            from official calc_dcd where n_lambda is an exponent on count).

    Returns:
        scalar (batch mean of weighted distances)
    """
    B, N, _ = src.shape

    _, idx = safe_knn(src, tgt, 1)  # (B, N, 1)
    idx = idx.reshape(B, N)

    bi = jt.arange(B).reshape(-1, 1)  # (B, 1)
    nn_tgt = tgt[bi, idx]  # (B, N, 3)

    d = ((src - nn_tgt) ** 2).sum(dim=-1)  # (B, N) squared distances

    exp_d = jt.exp(-d / alpha)  # (B, N)
    exp_d_mean = exp_d.mean(dim=-1, keepdims=True)  # (B, 1)

    w = (1.0 - n_lambda) * exp_d / (exp_d_mean + 1e-12) + n_lambda  # (B, N)

    weighted_d = (d / (w + 1e-12)).mean()  # scalar

    return weighted_d


def compute_dcd_like_loss(
    pred: jt.Var,
    clean: jt.Var,
    alpha: float = 1000.0,
    n_lambda: float = 0.5,
) -> jt.Var:
    """计算早期密度加权 Chamfer 对照损失。

    pred 和 clean 为 (B, N, 3) 点云，返回可求导的标量。
    alpha 是 exp(-d / alpha) 的温度，n_lambda 是权重线性混合系数；
    默认值用于保留历史对照设置，不对应官方 DCD 的参数定义。
    """
    fwd = _one_side_dcd_like(pred, clean, alpha, n_lambda)
    bwd = _one_side_dcd_like(clean, pred, alpha, n_lambda)
    return 0.5 * (fwd + bwd)
