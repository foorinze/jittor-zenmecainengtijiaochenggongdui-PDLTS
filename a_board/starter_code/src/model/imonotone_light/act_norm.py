"""单调可逆层使用的 ActNorm（激活归一化）。

来源：PD-LTS 的 models/layers/act_norm.py。
首次正向使用输入统计量初始化平移和缩放，之后使用模块参数。
初始化通过 update 写入已分离梯度的统计量，不替换参数对象。
方差下限为 0.2，与上游实现一致，避免近零方差造成过大的指数缩放。

初始化只分离输入和统计量，不用 no_grad 包裹参数更新，以保留参数梯度。
"""

from __future__ import annotations

import jittor as jt
from jittor import nn

# Variance lower bound per PyTorch PD-LTS reference (act_norm.py).
# Without this, ActNorm init can produce exp(weight) ~ 1000+ when channels
# have near-zero batch variance, immediately destabilizing forward & backward.
# Near-zero variance without this bound can destabilize initialization;
# the bound limits the initial exponential scale.
ACTNORM_VAR_MIN = 0.2


class ActNorm(nn.Module):

    def __init__(self, channel: int):
        super().__init__()
        # bias/weight 是 trainable 参数 (default requires_grad=True in Jittor)
        self.bias = jt.zeros((1, 1, channel))
        self.weight = jt.zeros((1, 1, channel))
        # is_inited 是 buffer, 不参与训练
        self.is_inited = jt.zeros((1,)).stop_grad()

    def _initialize(self, x: jt.Var):
        # Detach the input statistics while keeping parameter updates trainable.
        flat = x.detach().reshape(-1, x.shape[-1])
        mean = flat.mean(0).reshape(1, 1, -1)
        var = flat.var(0).reshape(1, 1, -1)
        # 方差下限设为 ACTNORM_VAR_MIN 限制初始 exp(weight)
        var = jt.maximum(var, jt.array(ACTNORM_VAR_MIN))
        # .update(...detach()) 写值：保留参数注册和梯度可训练性
        self.bias.update((-mean).detach())
        self.weight.update((-0.5 * jt.log(var + 1e-6)).detach())
        self.is_inited.update(jt.ones((1,)))

    def execute(self, x: jt.Var, logpx: jt.Var | None = None):
        if float(self.is_inited.item()) < 0.5:
            self._initialize(x)

        y = (x + self.bias) * jt.exp(self.weight)

        if logpx is None:
            return y
        else:
            ldj = self.weight.sum() * jt.ones((x.shape[0],))
            return y, logpx + ldj.reshape(-1, 1)

    def inverse(self, y: jt.Var, logpy: jt.Var | None = None):
        x = y * jt.exp(-self.weight) - self.bias

        if logpy is None:
            return x
        else:
            ldj = self.weight.sum() * jt.ones((y.shape[0],))
            return x, logpy - ldj.reshape(-1, 1)
