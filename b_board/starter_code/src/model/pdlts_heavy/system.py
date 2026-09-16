"""PDLTS Heavy ModelSpec wrapper.

Simplified version of PDLTSLight system.py for Heavy smoke probe.
Only chamfer + fixed_L2 loss (DCD is not used).
"""

from typing import Dict, List

import jittor as jt
import numpy as np

from ..spec import ModelSpec
from ...data.asset import Asset
from ..pdlts_light.layer import safe_knn
from .model import PDLTSHeavyNetwork


LOSS_KEY_CHAMFER = "chamfer"
LOSS_KEY_L2 = "l2"


def _chamfer_l2_breakdown(x: jt.Var, y: jt.Var):
    _, idx_x2y = safe_knn(x, y, 1)
    _, idx_y2x = safe_knn(y, x, 1)
    B, N, _ = x.shape
    _, M, _ = y.shape

    bi = jt.arange(B).view(-1, 1)
    idx_x2y_BN = idx_x2y.reshape(B, N)
    nn_y = y[bi, idx_x2y_BN]
    nn_x = x[bi, idx_y2x.reshape(B, M)]

    d_x2y = ((x - nn_y) ** 2).sum(dim=-1)
    d_y2x = ((y - nn_x) ** 2).sum(dim=-1)
    chamfer = d_x2y.mean() + d_y2x.mean()
    return chamfer, d_x2y, d_y2x, idx_x2y_BN, nn_y


class PDLTSHeavy(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config

        self.network = PDLTSHeavyNetwork(
            pc_channel=cfg.get("pc_channel", 3),
            aug_channel=cfg.get("aug_channel", 32),
            n_injector=cfg.get("n_injector", 10),
            cut_channel=cfg.get("cut_channel", 16),
            nflow_module=cfg.get("nflow_module", 10),
            num_neighbors=cfg.get("num_neighbors", 32),
            mlgc_hidden=cfg.get("mlgc_hidden", 64),
            inn_nhidden=cfg.get("inn_nhidden", 2),
            inn_idim=cfg.get("inn_idim", 64),
            inn_coeff=cfg.get("inn_coeff", 0.9),
            inn_n_iterations=cfg.get("inn_n_iterations", None),
            inn_sn_atol=cfg.get("inn_sn_atol", 1e-3),
            inn_sn_rtol=cfg.get("inn_sn_rtol", 1e-3),
            geom_p=cfg.get("geom_p", 0.5),
            n_exact_terms=cfg.get("n_exact_terms", 0),
            neumann_grad=cfg.get("neumann_grad", True),
            activation_fn=cfg.get("activation_fn", "elu"),
        )

        self._last_train_metrics: Dict[str, float] = {}

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                d = {
                    "pc_noisy": b.meta["pc_noisy"].astype(np.float32),
                    "pc_clean": b.meta["pc_clean"].astype(np.float32),
                }
            else:
                d = {"pc_noisy": b.sampled_vertices_noisy.astype(np.float32)}
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices.astype(np.float32)
            res.append(d)
        return res

    def training_step(self, batch: Dict) -> Dict:
        pc_noisy = batch["pc_noisy"]
        pc_clean = batch["pc_clean"]
        patch_size = pc_noisy.shape[-2]
        pc_noisy_flat = pc_noisy.reshape(-1, patch_size, 3)
        pc_clean_flat = pc_clean.reshape(-1, patch_size, 3)

        denoised, _ldj, _loss_d = self.network(pc_noisy_flat)

        chamfer, d_x2y, d_y2x, idx_x2y_BN, nn_y = _chamfer_l2_breakdown(
            denoised, pc_clean_flat
        )
        l2 = ((denoised - pc_clean_flat) ** 2).sum(dim=-1).mean()

        out = {
            LOSS_KEY_CHAMFER: chamfer,
            LOSS_KEY_L2: l2,
        }

        self._last_train_metrics = {
            "L_chamfer_raw": chamfer,
            "L_l2_raw": l2,
        }

        return out

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        from ..pdlts_light.denoise import denoise_full_cloud

        pc_noisy_batch = batch["pc_noisy"]
        patch_size = int(self.model_config.get("patch_size", 1024))
        seed_k = int(self.model_config.get("predict_seed_k", 3))
        seed_k_alpha = int(self.model_config.get("predict_seed_k_alpha", 30))

        res = []
        for i in range(pc_noisy_batch.shape[0]):
            pc_noisy = pc_noisy_batch[i]
            pc_noisy_np = pc_noisy.numpy().astype(np.float32)
            denoised_np, coverage_info = denoise_full_cloud(
                self.network,
                pc_noisy_np,
                patch_size=patch_size,
                seed_k=seed_k,
                seed_k_alpha=seed_k_alpha,
                return_coverage=True,
            )
            assert denoised_np.shape == pc_noisy_np.shape
            res.append({"pc_denoised": denoised_np, "coverage_info": coverage_info})
        return res
