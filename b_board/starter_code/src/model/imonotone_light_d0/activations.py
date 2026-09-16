"""早期基线对照：LipSwish 激活（可学习 beta），源实现一致地对齐 PyTorch 基线。

PyTorch 源：上游 PyTorch 参考实现/models/layers/base/activations.py
`class Swish`（早期训练脚本的 `activation_fn="swish"` 实际调用的就是这个类，
见 `models/model_light/deflow.py:28: 'swish': lambda b: base_layers.Swish()`）。

forward:  swish(x) = (x * sigmoid(x * softplus(beta))) / 1.1
grad:     d/dx swish(x) = (sigmoid(bx) + bx*sigmoid(bx)*(1-sigmoid(bx))) / 1.1
          其中 bx = x * softplus(beta)

除以 1.1 是 Lipschitz-1 的软约束（LipSwish 论文里的常数），beta 初始化为 0.5，
是可学习参数（每个 FCNet 层用一个独立 beta，早期 checkpoint 里对应 60 个
`*.beta` shape=[1] 的参数——3 层 nhidden=2 的 FCNet 共 2 个 preact/非-preact
nnet，每个 nnet 3 个激活位置，12 个 FlowAssembly x 2 iMonotoneBlock = 24 个
FCNet，每个 FCNet 有 preact(0或1) + nhidden(2) 个激活 = 2或3 个 beta，
24 FCNet 中 12 个 preact=False（2 个 beta）+ 12 个 preact=True（3 个 beta）
= 12*2 + 12*3 = 60，与 checkpoint 实测吻合。
"""

from __future__ import annotations

import jittor as jt
from jittor import nn


class LipSwish(nn.Module):
    """d 可学习 beta 的 LipSwish 激活。逐层独立实例，beta 从 checkpoint 加载。"""

    def __init__(self):
        super().__init__()
        self.beta = jt.array([0.5]).float32()

    def _softplus_beta(self) -> jt.Var:
        # jt.nn.softplus 默认 beta=1, threshold=20，与 PyTorch F.softplus 默认一致
        return nn.softplus(self.beta, beta=1, threshold=20)

    def execute(self, x: jt.Var) -> jt.Var:
        sp_beta = self._softplus_beta()
        return (x * jt.sigmoid(x * sp_beta)) / 1.1

    def grad_wrt_input(self, x: jt.Var) -> jt.Var:
        """逐元素解析导数（不走 autograd），用于 JVP 网络构造。

        对应 PyTorch `Swish.grad(self, x)`。
        """
        sp_beta = self._softplus_beta()
        bx = x * sp_beta
        s = jt.sigmoid(bx)
        return (s + bx * s * (1.0 - s)) / 1.1

    def build_clone(self) -> "LipSwish":
        """no-grad 深拷贝，用于 iMonotoneBlock 的 root_find warm-up 副本。"""
        clone = LipSwish()
        clone.beta = self.beta.detach().clone()
        return clone
