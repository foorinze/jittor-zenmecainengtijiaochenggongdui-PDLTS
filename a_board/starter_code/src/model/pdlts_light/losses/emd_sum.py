"""EMD（地球移动距离）对照损失，未用于最终提交。

通过 emd_jittor/emd_op.py 调用 Jittor CUDA 核函数，返回每点平方距离
dist 的总和。task（任务）配置中的 emd 权重在外层施加；
权重为 0.1 时，损失计算式为 0.1 * dist.sum()，对应 PD-LTS
metric/loss.py::EarthMoverDistance.forward 的求和方式。

只对预测坐标求梯度，不对目标坐标求梯度；要求 batch <= 512 且
点数为 1024 的倍数。auction（拍卖）算法提供近似匹配，
同时返回匹配索引唯一率用于检查重复指派。
"""

from __future__ import annotations

import numpy as np
import jittor as jt

from ...emd_jittor import earth_mover_distance


def compute_emd_loss(pred: jt.Var, clean: jt.Var,
                     eps: float = 0.005, iters: int = 50):
    """EMD loss + assignment 唯一率指标。

    Args:
        pred:  (B, N, 3) 去噪 patch（有梯度），N % 1024 == 0，B ≤ 512
        clean: (B, N, 3) GT patch（无梯度）
        eps / iters: auction 算法参数，沿用原实现默认值

    Returns:
        loss:    标量 jt.Var = dist.sum()（0.1 系数在 task yaml 权重侧）
        metrics: dict，含 emd_assign_uniq（近似双射 1-1 占比，逐 patch
                 unique(assignment)/N 的均值）与 emd_dist_mean（每点平方
                 距离均值，日志直读用）
    """
    dist, assignment = earth_mover_distance(pred, clean, eps=eps, iters=iters)
    loss = dist.sum()

    # assignment / dist 的观测指标直接 .numpy()/.item() 取值（Jittor stop_grad()
    # 原地生效，若误用在 dist 上会把 loss 的梯度通路一起切断，见 uniformcd.py 注）。
    assign_np = assignment.numpy()                          # (B, N) int32
    B, N = assign_np.shape
    uniq = [np.unique(assign_np[b]).size / N for b in range(B)]
    metrics = {
        "emd_assign_uniq": float(np.mean(uniq)),
        "emd_dist_mean": float(dist.mean().item()),
    }
    return loss, metrics
