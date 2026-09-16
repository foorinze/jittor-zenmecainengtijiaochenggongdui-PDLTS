"""早期基线对照：源实现一致的前向组件（coeff=0.98/LipSwish）+ 已验证的
warm-started Banach 反传（可训练版 iMonotoneBlock）。

D0 的 `monotone_block.py::iMonotoneBlockD0` 只做前向推理（RootFind 无梯度，
`nnet.build_clone()` 冻结权重），不能训练。Q3a 需要反向可训性，因此本模块
复用训练版 `imonotone_light/monotone_block.py::_unrolled_banach`（warm-start
加有限步展开 Banach，已验证能支撑完整训练），套在 D0
的 source-exact FCNet（`fcnet.py::FCNetD0`，coeff=0.98 + 可学习 LipSwish）上。

与训练版 `iMonotoneBlock`（`imonotone_light/monotone_block.py`）的
唯一区别：nnet 换成 D0 的 `FCNetD0`（激活函数 LipSwish 而非 ELU，coeff 默认
0.98 而非 0.9）。反传算法逻辑完全不变，直接 import 复用，不复制代码。
"""

from __future__ import annotations

import math

import jittor as jt
from jittor import nn

from ..imonotone_light.monotone_block import _unrolled_banach, DEFAULT_UNROLL_STEPS
from ..imonotone_light.solvers import root_find

SQRT2 = math.sqrt(2)


class iMonotoneBlockD0Trainable(nn.Module):
    """D0 source-exact 前向组件 + warm-started Banach 反传，可训练版本。

    Forward:  y = sqrt(2)*w - x,  w = warm_start(root_find) + unroll_steps 个 Banach 迭代
    Inverse:  x = sqrt(2)*w - y,  同上，sign 相反

    unroll_steps 默认复用 历史实现的 DEFAULT_UNROLL_STEPS=5（历史实现已验证的训练期精度/成本平衡）。
    """

    def __init__(self, nnet, unroll_steps: int = DEFAULT_UNROLL_STEPS):
        super().__init__()
        self.nnet = nnet
        self.unroll_steps = unroll_steps

    def execute(self, x: jt.Var, logpx: jt.Var | None = None):
        nnet_clone = self.nnet.build_clone()
        x_detached = x.detach()
        scaled_x_detached = SQRT2 * x_detached
        w_warm = root_find(lambda z: nnet_clone(z), scaled_x_detached)
        w_warm = w_warm.detach()

        scaled_x = SQRT2 * x
        w = _unrolled_banach(self.nnet, scaled_x, w_warm, steps=self.unroll_steps, sign=-1)

        y = SQRT2 * w - x

        if logpx is None:
            return y
        raise NotImplementedError("Q3a 只做纯 Chamfer 训练，不需要 logdet 路径")

    def inverse(self, y: jt.Var, logpy: jt.Var | None = None):
        nnet_clone = self.nnet.build_clone()
        y_detached = y.detach()
        scaled_y_detached = SQRT2 * y_detached
        w_warm = root_find(lambda z: -nnet_clone(z), scaled_y_detached)
        w_warm = w_warm.detach()

        scaled_y = SQRT2 * y
        w = _unrolled_banach(self.nnet, scaled_y, w_warm, steps=self.unroll_steps, sign=+1)

        x = SQRT2 * w - y

        if logpy is None:
            return x
        raise NotImplementedError("Q3a 只做纯 Chamfer 训练，不需要 logdet 路径")
