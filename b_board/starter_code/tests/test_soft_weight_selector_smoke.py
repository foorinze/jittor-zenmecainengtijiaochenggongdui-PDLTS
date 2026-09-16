"""Soft-weight selector smoke test.

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_soft_weight_selector_smoke

验证目标:
    1. keep_head / weighted_y2x / keep_reg 能进入 training_step。
    2. _last_train_metrics 含 keep-weight 诊断字段。
    3. weighted_y2x + keep_reg 只更新 keep head，不回推主干。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt

from src.model.parse import get_model
from src.model.pdlts_light.system import (
    LOSS_KEY_CHAMFER,
    LOSS_KEY_KEEP_REG,
    LOSS_KEY_L2,
    LOSS_KEY_WEIGHTED_Y2X,
)


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


def _softw_model():
    cfg = {
        "__target__": "PDLTSLight",
        "pc_channel": 3,
        "aug_channel": 48,
        "n_injector": 12,
        "cut_channel": 24,
        "nflow_module": 12,
        "num_neighbors": 8,
        "mlgc_hidden": 32,
        "coupling_hidden": 32,
        "log_scale_clamp": 0.1,
        "target_mode": "paired_idx",
        "l2_target_mode": "fixed",
        "keep_head_mode": "soft_weight",
        "keep_head_hidden": 32,
        "keep_loss_mode": "soft_weight",
        "keep_target_keep": 0.67,
        "keep_var_floor": 0.01,
        "keep_var_lambda": 1.0,
    }
    return get_model(model_config=cfg, transform_config={})


def _batch(B=2, P=1, M=32):
    return {
        "pc_noisy": jt.randn(B, P, M, 3),
        "pc_clean": jt.randn(B, P, M, 3),
    }


def test_softw_training_step_emits_losses_and_metrics():
    model = _softw_model()
    out = model.training_step(_batch())

    for key in (LOSS_KEY_CHAMFER, LOSS_KEY_L2, LOSS_KEY_WEIGHTED_Y2X, LOSS_KEY_KEEP_REG):
        assert key in out, f"missing loss key: {key}"
        assert np.isfinite(float(out[key].item())), f"non-finite loss for {key}"

    metrics = getattr(model, "_last_train_metrics", {})
    expected = {
        "weighted_y2x_raw",
        "keep_reg_raw",
        "w_mean",
        "w_var",
        "w_std",
        "w_entropy",
        "w_resp_mean",
        "keep_boundary_ratio",
    }
    missing = sorted(expected - set(metrics.keys()))
    assert not missing, f"missing keep metrics: {missing}"
    print("[PASS] test_softw_training_step_emits_losses_and_metrics")


def test_softw_keep_loss_only_updates_keep_head():
    model = _softw_model()
    out = model.training_step(_batch())
    keep_loss = out[LOSS_KEY_WEIGHTED_Y2X] + out[LOSS_KEY_KEEP_REG]

    keep_named = []
    trunk_named = []
    for name, param in model.named_parameters():
        if name.endswith("running_mean") or name.endswith("running_var"):
            continue
        if name.endswith("is_inited") or name.endswith("mask"):
            continue
        if name.endswith("channel_mask"):
            continue
        if "MLP_keep" in name and (name.endswith("weight") or name.endswith("bias")):
            keep_named.append((name, param))
        elif "MLP_keep" not in name and name.endswith("weight"):
            trunk_named.append((name, param))

    assert keep_named, "no keep-head weight found"
    assert trunk_named, "no trunk weight found"

    trunk_name, trunk_param = trunk_named[0]
    grad_targets = [param for _, param in keep_named] + [trunk_param]
    grads = jt.grad(keep_loss, grad_targets)

    keep_grad_sums = []
    for (keep_name, _), grad in zip(keep_named, grads[:-1]):
        if grad is None:
            keep_grad_sums.append((keep_name, 0.0))
        else:
            keep_grad_sums.append((keep_name, float(jt.abs(grad).sum().item())))
    best_keep_name, best_keep_grad_sum = max(keep_grad_sums, key=lambda item: item[1])
    assert best_keep_grad_sum > 1e-12, (
        f"all keep-head grads are zero: {keep_grad_sums}"
    )

    trunk_grad = grads[-1]
    if trunk_grad is None:
        trunk_grad_sum = 0.0
    else:
        trunk_grad_sum = float(jt.abs(trunk_grad).sum().item())
    assert trunk_grad_sum < 1e-12, (
        f"keep loss leaked into trunk {trunk_name}: grad_sum={trunk_grad_sum:.3e}"
    )
    print(
        "[PASS] test_softw_keep_loss_only_updates_keep_head "
        f"(active_keep={best_keep_name}, grad_sum={best_keep_grad_sum:.3e})"
    )


if __name__ == "__main__":
    _setup()
    test_softw_training_step_emits_losses_and_metrics()
    test_softw_keep_loss_only_updates_keep_head()
    print("[PASS] soft-weight selector smoke complete")
