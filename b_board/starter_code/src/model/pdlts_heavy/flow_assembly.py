"""Heavy FlowAssembly: [iMonotoneBlock, ActNorm, iMonotoneBlock(preact), ActNorm].

Drop-in replacement for Light's FlowAssembly (AffineCoupling-based).
Interface: execute(x, logpx=None) / inverse(y, logpy=None).
"""

from __future__ import annotations

import jittor as jt
from jittor import nn

from .monotone_block import iMonotoneBlock
from .fcnet import FCNet
from .act_norm import ActNorm


class HeavyFlowAssembly(nn.Module):

    def __init__(
        self,
        channel: int,
        nhidden: int = 2,
        idim: int = 64,
        coeff: float = 0.9,
        n_iterations: int | None = None,
        sn_atol: float = 1e-3,
        sn_rtol: float = 1e-3,
        geom_p: float = 0.5,
        n_exact_terms: int = 0,
        neumann_grad: bool = True,
        activation_fn: str = "elu",
    ):
        super().__init__()

        nnet1 = FCNet(
            channel=channel, preact=False, nhidden=nhidden, idim=idim,
            coeff=coeff, n_iterations=n_iterations,
            sn_atol=sn_atol, sn_rtol=sn_rtol,
            activation_fn=activation_fn,
        )
        nnet2 = FCNet(
            channel=channel, preact=True, nhidden=nhidden, idim=idim,
            coeff=coeff, n_iterations=n_iterations,
            sn_atol=sn_atol, sn_rtol=sn_rtol,
            activation_fn=activation_fn,
        )

        self.layers = nn.ModuleList([
            iMonotoneBlock(nnet1, geom_p=geom_p, n_exact_terms=n_exact_terms,
                           neumann_grad=neumann_grad),
            ActNorm(channel),
            iMonotoneBlock(nnet2, geom_p=geom_p, n_exact_terms=n_exact_terms,
                           neumann_grad=neumann_grad),
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
