"""PDLTS Heavy network assembly (MLGC + Heavy INN + FBM).

Uses the same MLGC feature extraction as Light but with Heavy's channel tables
(n_injector=10, aug_channel=32, cut_channel=16) and HeavyFlowAssembly
(iMonotoneBlock-based) instead of Light's AffineCoupling FlowAssembly.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import jittor as jt
from jittor import nn

from ..pdlts_light.layer import safe_knn

from ..pdlts_light.layer import EdgeConv, FeatMergeUnit, PreConv, noiseEdgeConv
from .flow_assembly import HeavyFlowAssembly

HEAVY_AUG_CHANNEL = 32
HEAVY_N_INJECTOR = 10
HEAVY_CUT_CHANNEL = 16
HEAVY_NFLOW_MODULE = 10
HEAVY_NUM_NEIGHBORS = 32

_IN_CHANNEL_E = [16, 48, 80, 112, 144, 176, 96, 120, 144, 168]
_IN_CHANNEL_A = [48, 80, 112, 144, 176, 96, 120, 144, 168, 96]
_OUT_CHANNEL = [32, 32, 32, 32, 32, 96, 24, 24, 24, 96]
_CONCAT_FALSE_INDEX = {5, 9}
_MLGC_HIDDEN = 64


class PDLTSHeavyNetwork(nn.Module):

    def __init__(
        self,
        pc_channel: int = 3,
        aug_channel: int = HEAVY_AUG_CHANNEL,
        n_injector: int = HEAVY_N_INJECTOR,
        cut_channel: int = HEAVY_CUT_CHANNEL,
        nflow_module: int = HEAVY_NFLOW_MODULE,
        num_neighbors: int = HEAVY_NUM_NEIGHBORS,
        mlgc_hidden: int = _MLGC_HIDDEN,
        inn_nhidden: int = 2,
        inn_idim: int = 64,
        inn_coeff: float = 0.9,
        inn_n_iterations: int | None = None,
        inn_sn_atol: float = 1e-3,
        inn_sn_rtol: float = 1e-3,
        geom_p: float = 0.5,
        n_exact_terms: int = 0,
        neumann_grad: bool = True,
        activation_fn: str = "elu",
    ):
        super().__init__()

        assert n_injector == len(_IN_CHANNEL_E)
        assert nflow_module >= n_injector

        self.pc_channel = pc_channel
        self.aug_channel = aug_channel
        self.n_injector = n_injector
        self.cut_channel = cut_channel
        self.nflow_module = nflow_module
        self.num_neighbors = num_neighbors

        inj_channel = pc_channel + aug_channel  # 35

        # MLGC
        self.noise_params = noiseEdgeConv(
            in_channel=pc_channel, hidden_channel=32, out_channel=aug_channel
        )
        self.PreConv = PreConv(in_channel=pc_channel, out_channel=16)
        self.feat_Conv = nn.ModuleList()
        self.AdaptConv = nn.ModuleList()
        for i in range(n_injector):
            concat = i not in _CONCAT_FALSE_INDEX
            self.feat_Conv.append(EdgeConv(
                in_channel=_IN_CHANNEL_E[i],
                hidden_channel=mlgc_hidden,
                out_channel=_OUT_CHANNEL[i],
                concat=concat,
            ))
            self.AdaptConv.append(FeatMergeUnit(
                in_channel=_IN_CHANNEL_A[i],
                hidden_channel=mlgc_hidden,
                out_channel=inj_channel,
            ))

        # Heavy INN
        self.flow_assemblies = nn.ModuleList([
            HeavyFlowAssembly(
                channel=inj_channel,
                nhidden=inn_nhidden,
                idim=inn_idim,
                coeff=inn_coeff,
                n_iterations=inn_n_iterations,
                sn_atol=inn_sn_atol,
                sn_rtol=inn_sn_rtol,
                geom_p=geom_p,
                n_exact_terms=n_exact_terms,
                neumann_grad=neumann_grad,
                activation_fn=activation_fn,
            )
            for _ in range(nflow_module)
        ])

        # FBM mask
        mask = np.ones((1, 1, inj_channel), dtype=np.float32)
        mask[..., -cut_channel:] = 0.0
        self.channel_mask = jt.array(mask).stop_grad()

    def unit_coupling(self, xyz: jt.Var, knn_idx: jt.Var) -> jt.Var:
        return self.noise_params(xyz, knn_idx)

    def feat_extract(self, xyz: jt.Var, knn_idx: jt.Var) -> List[jt.Var]:
        cs: List[jt.Var] = []
        f = self.PreConv(xyz, knn_idx)
        for i in range(self.n_injector):
            f = self.feat_Conv[i](f, knn_idx)
            inj_f = self.AdaptConv[i](f)
            cs.append(inj_f)
        return cs

    def f(self, x: jt.Var, inj_f: List[jt.Var]) -> Tuple[jt.Var, jt.Var]:
        B = x.shape[0]
        log_det_J = jt.zeros((B,))
        for i in range(self.nflow_module):
            if i < self.n_injector:
                x = x + inj_f[i]
            x = self.flow_assemblies[i](x)
        return x, log_det_J

    def g(self, z: jt.Var, inj_f: List[jt.Var]) -> jt.Var:
        for i in reversed(range(self.nflow_module)):
            z = self.flow_assemblies[i].inverse(z)
            if i < self.n_injector:
                z = z - inj_f[i]
        return z

    def execute(self, xyz: jt.Var) -> Tuple[jt.Var, jt.Var, jt.Var]:
        B, N, _ = xyz.shape
        _, knn_idx = safe_knn(xyz, xyz, self.num_neighbors)
        inj_f = self.feat_extract(xyz, knn_idx)
        aug = self.unit_coupling(xyz, knn_idx)
        x = jt.concat([xyz, aug], dim=-1)
        z, ldj = self.f(x, inj_f)
        predict_z = z * self.channel_mask
        full_x = self.g(predict_z, inj_f)
        denoised = full_x[..., :self.pc_channel]
        loss_denoise = jt.zeros((1,))
        return denoised, ldj, loss_denoise
