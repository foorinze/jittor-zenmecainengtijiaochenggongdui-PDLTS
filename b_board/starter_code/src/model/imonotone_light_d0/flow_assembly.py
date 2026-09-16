"""早期基线对照：源实现一致的 IMonotoneFlowAssembly（coeff=0.98/LipSwish/RootFind）。

结构对齐 PyTorch `models/model_light/deflow.py:337-343`：
    [iMonotoneBlockD0(nnet1, preact=False), ActNorm,
     iMonotoneBlockD0(nnet2, preact=True), ActNorm]

`ActNorm` 复用基础 `imonotone_light.act_norm.ActNorm`（该实现与
PyTorch `models/layers/act_norm.py` 已逐字段对齐，coeff 不影响 ActNorm，
不需要 D0 专属版本）。
"""

from __future__ import annotations

import jittor as jt
from jittor import nn

from ..imonotone_light.act_norm import ActNorm
from .fcnet import FCNetD0
from .monotone_block import iMonotoneBlockD0


class IMonotoneFlowAssemblyD0(nn.Module):

    def __init__(
        self,
        channel: int,
        nhidden: int = 2,
        idim: int = 64,
        coeff: float = 0.98,
        n_iterations: int | None = None,
        sn_atol: float = 1e-3,
        sn_rtol: float = 1e-3,
        root_find_atol: float = 1e-6,
        root_find_rtol: float = 1e-6,
        root_find_max_iter: int = 10,
    ):
        super().__init__()

        nnet1 = FCNetD0(
            channel=channel, preact=False, nhidden=nhidden, idim=idim,
            coeff=coeff, n_iterations=n_iterations,
            sn_atol=sn_atol, sn_rtol=sn_rtol,
        )
        nnet2 = FCNetD0(
            channel=channel, preact=True, nhidden=nhidden, idim=idim,
            coeff=coeff, n_iterations=n_iterations,
            sn_atol=sn_atol, sn_rtol=sn_rtol,
        )

        self.layers = nn.ModuleList([
            iMonotoneBlockD0(nnet1, atol=root_find_atol, rtol=root_find_rtol,
                              max_iter=root_find_max_iter),
            ActNorm(channel),
            iMonotoneBlockD0(nnet2, atol=root_find_atol, rtol=root_find_rtol,
                              max_iter=root_find_max_iter),
            ActNorm(channel),
        ])

    def execute(self, x: jt.Var, logpx: jt.Var | None = None):
        for layer in self.layers:
            if logpx is None:
                x = layer(x)
            else:
                x, logpx = layer(x, logpx)
        if logpx is None:
            return x
        return x, logpx

    def inverse(self, y: jt.Var, logpy: jt.Var | None = None):
        for layer in reversed(self.layers):
            if logpy is None:
                y = layer.inverse(y)
            else:
                y, logpy = layer.inverse(y, logpy)
        if logpy is None:
            return y
        return y, logpy

    def root_find_diagnostics(self) -> list[dict]:
        """Q1 附带必测：收集本 FlowAssembly 内两个 iMonotoneBlockD0 最近一次 root_find
        的收敛步数与残差。"""
        diags = []
        for layer in self.layers:
            if isinstance(layer, iMonotoneBlockD0):
                diags.append({
                    "iters": layer.last_root_find_iters,
                    "residual": layer.last_root_find_residual,
                })
        return diags
