"""早期基线对照：源实现一致的 iMonotoneBlock（RootFind 不动点求解，无近似展开）。

PyTorch 源：上游 PyTorch 参考实现/models/layers/iMonotoneBlock.py:51-91
（forward/inverse），上游 PyTorch 参考实现/models/layers/solvers.py
（RootFind.banach_find_root / find_fixed_point / MonotoneBlockBackward）。

D0 只做**前向数值对照**，不训练、不需要反传。这让本实现比训练版
（`imonotone_light.monotone_block.iMonotoneBlock`，用 warm-start 加有限步展开
Banach 近似训练期梯度）更简单：

    PyTorch forward 的真实计算链（训练/推理通用）：
        nnet_copy = self.nnet.build_clone()               # 冻结权重副本
        w_value = RootFind(nnet_copy, sqrt2*x).detach()    # 定点迭代求解，无梯度
        w_proxy = sqrt2*x - self.nnet(w_value)             # 用于 backward 占位
        w = MonotoneBlockBackward.apply(nnet_copy, w_proxy, sqrt2*x)  # forward 恒等于 w_proxy
        y = sqrt2*w - x

    `MonotoneBlockBackward.forward` 只是 `return y`（占位，真正的隐式微分在
    `.backward()` 里，D0 推理不触发）。且 eval 模式下 `self.nnet(w_value)` 与
    `nnet_copy(w_value)` 权重完全相同（`InducedNormLinear.execute` 在
    `is_training()=False` 时也是 `update=False`），所以推理路径下
    `w_proxy` 在数值上就是收敛后的定点解 `w_value`。D0 直接实现这条完整链
    （不做代数简化），保持与 PyTorch 逐步骤对齐，方便逐层数值比较时定位差异层。

Solves the fixed-point equation forward: w + G(w) = sqrt(2)*x  ->  y = sqrt(2)*w - x
Solves the fixed-point equation inverse: w - G(w) = sqrt(2)*y  ->  x = sqrt(2)*w - y
"""

from __future__ import annotations

import math

import jittor as jt
from jittor import nn

SQRT2 = math.sqrt(2)


class iMonotoneBlockD0(nn.Module):

    def __init__(self, nnet, atol: float = 1e-6, rtol: float = 1e-6, max_iter: int = 10):
        super().__init__()
        self.nnet = nnet
        self.atol = atol
        self.rtol = rtol
        self.max_iter = max_iter
        # 诊断字段：最近一次 root_find 的收敛信息（Q1 附带必测：收敛步数与残差）
        self.last_root_find_iters: int | None = None
        self.last_root_find_residual: float | None = None

    def _root_find_with_diag(self, g_fn, y: jt.Var) -> jt.Var:
        """带收敛诊断的定点迭代（沿用 imonotone_light.solvers.find_fixed_point 的算法，
        但显式记录迭代步数和终止残差，供 D0 附带必测使用）。
        """
        with jt.no_grad():
            x = y - g_fn(y)
            n_iter = 0
            for i in range(self.max_iter):
                x_prev = x
                x = y - g_fn(x)
                n_iter = i + 1
                tol = self.atol + y.abs() * self.rtol
                residual = ((x - x_prev) ** 2 / (tol + 1e-12))
                if jt.all(residual < 1.0):
                    self.last_root_find_iters = n_iter
                    self.last_root_find_residual = float(residual.max().numpy().item())
                    return x.detach()
            # 未在 max_iter 内收敛：仍返回当前值，但记录未收敛状态
            tol = self.atol + y.abs() * self.rtol
            residual = ((x - x_prev) ** 2 / (tol + 1e-12))
            self.last_root_find_iters = n_iter
            self.last_root_find_residual = float(residual.max().numpy().item())
        return x.detach()

    def execute(self, x: jt.Var, logpx: jt.Var | None = None):
        nnet_clone = self.nnet.build_clone()
        x0 = x.detach()
        scaled_x0 = SQRT2 * x0

        w_value = self._root_find_with_diag(lambda z: nnet_clone(z), scaled_x0)

        # w_proxy：源码对齐步骤，推理模式下数值上等于收敛后的 w_value
        # （self.nnet 与 nnet_clone 在 eval 模式权重完全一致）。
        scaled_x = SQRT2 * x
        w_proxy = scaled_x - self.nnet(w_value)
        w = w_proxy  # MonotoneBlockBackward.forward 在推理路径下的等价行为

        y = SQRT2 * w - x

        if logpx is None:
            return y
        raise NotImplementedError("D0 只做前向推理对拍，不需要 logdet 路径")

    def inverse(self, y: jt.Var, logpy: jt.Var | None = None):
        nnet_clone = self.nnet.build_clone()
        y0 = y.detach()
        scaled_y0 = SQRT2 * y0

        w_value = self._root_find_with_diag(lambda z: -nnet_clone(z), scaled_y0)

        scaled_y = SQRT2 * y
        w_proxy = scaled_y + self.nnet(w_value)
        w = w_proxy

        x = SQRT2 * w - y

        if logpy is None:
            return x
        raise NotImplementedError("D0 只做前向推理对拍，不需要 logdet 路径")
