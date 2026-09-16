"""Jittor 可逆单调块，使用不动点求解和近似梯度。

来源：PD-LTS 的 models/layers/iMonotoneBlock.py。
正向：w + G(w) = sqrt(2)*x，y = sqrt(2)*w - x。
逆向：w - G(w) = sqrt(2)*y，x = sqrt(2)*w - y。
对数行列式使用 Neumann（诺伊曼）级数近似，Heavy 训练损失不使用该项。

先在分离梯度的网络副本上求不动点，再从该结果展开 5 步
Banach（不动点）迭代，保留输入及网络参数的梯度。
有限步展开提供近似反向传播，没有实现原 PyTorch 版本中求解
(I + J_g^T) z = grad 的完整隐式微分。
展开后的结果不等于无限迭代极限；收缩性和正逆向误差需要分别检查。

Jittor 的 stop_grad() 会修改当前变量的标志。分离副本应使用 detach()，
以免切断仍需训练的参数或输入的梯度。
"""

from __future__ import annotations

import math
import logging

import numpy as np
import jittor as jt
from jittor import nn

from .solvers import find_fixed_point, root_find

logger = logging.getLogger(__name__)

SQRT2 = math.sqrt(2)

# Unrolled Banach iteration steps for backward gradient surrogate.
# Higher = more accurate gradient at the cost of memory (each step adds to
# the autograd graph). 5 is a balance for density-loss smoke; can raise to 10 if
# grad coverage tests still indicate poor gradient flow.
DEFAULT_UNROLL_STEPS = 5


def geometric_sample(p: float, n_samples: int):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p: float, k: int, offset: int) -> float:
    if k <= offset:
        return 1.0
    k = k - offset
    return (1 - p) ** max(k - 1, 0)


def _unrolled_banach(nnet, scaled_input: jt.Var, w_init: jt.Var,
                     steps: int, sign: int) -> jt.Var:
    """Unrolled Banach iteration with autograd connected.

    Solves the fixed-point equation:
        forward (sign=-1): w = scaled - nnet(w)
        inverse (sign=+1): w = scaled + nnet(w)

    Starts from w_init (a no-grad approximation of the true fixed point) and
    runs `steps` iterations with autograd enabled. The result is differentiable
    w.r.t. both `scaled_input` and the parameters of `nnet`.

    Args:
        nnet: the Lipschitz network G
        scaled_input: sqrt(2)*x (forward) or sqrt(2)*y (inverse), with grad
        w_init: initial guess for w, typically the no-grad root_find result
        steps: number of Banach iterations (5 = default)
        sign: -1 for forward (w + G(w) = scaled), +1 for inverse (w - G(w) = scaled)
    """
    w = w_init
    for _ in range(steps):
        # Each iteration: w = scaled_input + sign * nnet(w)
        # autograd tracks both scaled_input and nnet params through this chain
        w = scaled_input + sign * nnet(w)
    return w


class iMonotoneBlock(nn.Module):

    def __init__(
        self,
        nnet,
        geom_p: float = 0.5,
        lamb: float = 2.0,
        n_power_series: int | None = None,
        n_dist: str = "geometric",
        n_samples: int = 1,
        n_exact_terms: int = 0,
        neumann_grad: bool = True,
        grad_in_forward: bool = False,
        unroll_steps: int = DEFAULT_UNROLL_STEPS,
    ):
        super().__init__()
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = geom_p
        self.lamb = lamb
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.n_exact_terms = n_exact_terms
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.unroll_steps = unroll_steps

    def execute(self, x: jt.Var, logpx: jt.Var | None = None):
        # === C-prime v2: warm-start unrolled Banach ===
        # CRITICAL: use .detach() not .stop_grad() on x. .stop_grad() modifies
        # the input Var's flag in-place, which when x is the output of an
        # upstream module silently cuts that module's gradient path.
        # Step 1: no-grad warm-up via root_find on detached clone (precision)
        nnet_clone = self.nnet.build_clone()
        x_detached = x.detach()
        scaled_x_detached = SQRT2 * x_detached
        w_warm = root_find(lambda z: nnet_clone(z), scaled_x_detached)
        w_warm = w_warm.detach()  # 无梯度初始点

        # Step 2: unrolled Banach with grad enabled
        scaled_x = SQRT2 * x  # 有梯度
        w = _unrolled_banach(self.nnet, scaled_x, w_warm,
                             steps=self.unroll_steps, sign=-1)

        y = SQRT2 * w - x

        if logpx is None:
            return y
        else:
            return y, logpx + self._logdetgrad(w)

    def inverse(self, y: jt.Var, logpy: jt.Var | None = None):
        nnet_clone = self.nnet.build_clone()
        y_detached = y.detach()
        scaled_y_detached = SQRT2 * y_detached
        w_warm = root_find(lambda z: -nnet_clone(z), scaled_y_detached)
        w_warm = w_warm.detach()

        scaled_y = SQRT2 * y
        # Inverse fixed-point: w = scaled_y + nnet(w), so sign=+1
        w = _unrolled_banach(self.nnet, scaled_y, w_warm,
                             steps=self.unroll_steps, sign=+1)

        x = SQRT2 * w - y

        if logpy is None:
            return x
        else:
            return x, logpy - self._logdetgrad(w)

    def _logdetgrad(self, w: jt.Var) -> jt.Var:
        """Neumann series logdet estimation with Hutchinson trace estimator.

        NOTE: not invoked in current PDLTSHeavy training_step (logpx never
        passed). Kept for source-alignment and future logdet enablement.
        """
        geom_p = self.geom_p
        n_samples_arr = geometric_sample(geom_p, self.n_samples)
        n_power_series = int(max(n_samples_arr)) + self.n_exact_terms

        def coeff_fn(k):
            return (1.0 / geometric_1mcdf(geom_p, k, self.n_exact_terms)
                    * sum(n_samples_arr >= k - self.n_exact_terms)
                    / len(n_samples_arr))

        def power_series_coeff_fn(k):
            return (-2.0) / k if k % 2 == 1 else 0.0

        vareps = jt.randn_like(w)

        w_grad = w.clone()
        w_grad.start_grad()
        g = self.nnet(w_grad)

        vjp = vareps
        logdetgrad = jt.zeros((w.shape[0],))

        for k in range(1, n_power_series + 1):
            vjp = jt.grad(g, w_grad, vjp)
            tr = (vjp.reshape(w.shape[0], -1) * vareps.reshape(w.shape[0], -1)).sum(1)
            delta = power_series_coeff_fn(k) * coeff_fn(k) * tr
            logdetgrad = logdetgrad + delta

        return logdetgrad.reshape(-1, 1)
