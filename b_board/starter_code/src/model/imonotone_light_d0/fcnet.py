"""早期基线对照：源实现一致的 FCNet（coeff=0.98 + LipSwish 可学习 beta）。

PyTorch 源：上游 PyTorch 参考实现/models/model_light/deflow.py:345-429
`class FCNet`（`densenet=False` 分支，早期训练脚本未启用 densenet）。

层序（严格对齐官方，nhidden=2 时）：
    preact=False: [Linear0(in->idim), Act1, Linear2(idim->idim), Act3, Linear4(idim->out)]
    preact=True:  [Act0, Linear1(in->idim), Act2, Linear3(idim->idim), Act4, Linear5(idim->out)]

`Linear*` = InducedNormLinear（复用基础实现中的 `imonotone_light.spectral_norm`，
该实现的 coeff/atol/rtol 本就是构造参数，D0 只需传 coeff=0.98 即可 source-exact，
不需要复制代码）。`Act*` = 本包 `LipSwish`（基础实现没有的可学习 beta
激活，是该对照实现新增的核心组件）。

state_dict key 对齐（PyTorch ep99 checkpoint 实测）：
    flow_assemblies.<i>.chain.0.nnet.nnet.{0,1,2,3,4}.*     (preact=False)
    flow_assemblies.<i>.chain.2.nnet.nnet.{0,1,2,3,4,5}.*   (preact=True)
本模块的 `self.layers` 顺序与上述数字下标严格一致，供权重转换脚本按位置映射。
"""

from __future__ import annotations

from typing import List

import jittor as jt
from jittor import nn

from ..imonotone_light.spectral_norm import InducedNormLinear
from .activations import LipSwish


class FCNetD0(nn.Module):

    def __init__(
        self,
        channel: int,
        preact: bool = False,
        nhidden: int = 2,
        idim: int = 64,
        coeff: float = 0.98,
        n_iterations: int | None = None,
        sn_atol: float = 1e-3,
        sn_rtol: float = 1e-3,
    ):
        super().__init__()
        self.preact = bool(preact)
        self.nhidden = int(nhidden)

        seq: List[nn.Module] = []
        last_dim_in = channel

        if self.preact:
            seq.append(LipSwish())

        for _ in range(nhidden):
            seq.append(InducedNormLinear(
                last_dim_in, idim, bias=True,
                coeff=coeff, n_iterations=n_iterations,
                atol=sn_atol, rtol=sn_rtol,
            ))
            seq.append(LipSwish())
            last_dim_in = idim

        seq.append(InducedNormLinear(
            last_dim_in, channel, bias=True,
            coeff=coeff, n_iterations=n_iterations,
            atol=sn_atol, rtol=sn_rtol,
        ))

        self.layers = nn.ModuleList(seq)

    def execute(self, x: jt.Var) -> jt.Var:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h

    def build_clone(self) -> "_FCNetD0Clone":
        """no-grad 深拷贝：InducedNormLinear -> nn.Linear（冻结权重），LipSwish -> LipSwish 副本。

        用于 iMonotoneBlockD0 的 RootFind 定点迭代（纯前向、无梯度）。
        """
        cloned_layers = []
        for layer in self.layers:
            if isinstance(layer, InducedNormLinear):
                cloned_layers.append(layer.build_clone())
            elif isinstance(layer, LipSwish):
                cloned_layers.append(layer.build_clone())
            else:
                raise TypeError(f"unexpected layer type in FCNetD0: {type(layer)}")
        return _FCNetD0Clone(cloned_layers)


class _FCNetD0Clone(nn.Module):
    """build_clone() 的返回类型：纯前向链，元素是 nn.Linear 或 LipSwish 副本。"""

    def __init__(self, layers_list):
        super().__init__()
        self.layers = nn.ModuleList(layers_list)

    def execute(self, x: jt.Var) -> jt.Var:
        h = x
        for layer in self.layers:
            h = layer(h)
        return h
