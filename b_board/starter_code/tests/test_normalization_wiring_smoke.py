"""验证 normalization 配置经 PDLTSLight 正式包装层真实生效。"""

from __future__ import annotations

import json

import jittor as jt
import numpy as np
from jittor import nn

from src.model.pdlts_light.system import PDLTSLight


def norm_layers(network) -> list[nn.Module]:
    layers = [network.PreConv.conv[1]]
    for edge in network.feat_Conv:
        layers.extend(conv[1] for conv in edge.convs)
    for adapt in network.AdaptConv:
        layers.extend(conv[1] for conv in adapt.convs)
    return layers


def build(mode: str) -> PDLTSLight:
    return PDLTSLight(
        {
            "pc_channel": 3,
            "aug_channel": 48,
            "n_injector": 12,
            "cut_channel": 24,
            "nflow_module": 12,
            "num_neighbors": 8,
            "mlgc_hidden": 32,
            "mlgc_norm_mode": mode,
            "mlgc_group_norm_max_groups": 8,
            "coupling_hidden": 32,
            "log_scale_clamp": 0.1,
        },
        {},
    )


def main() -> None:
    jt.flags.use_cuda = 1
    jt.set_global_seed(123)
    control = build("batch")
    jt.set_global_seed(123)
    candidate = build("group")

    control_norms = norm_layers(control.network)
    candidate_norms = norm_layers(candidate.network)
    assert len(control_norms) == len(candidate_norms) == 49
    assert all(type(layer).__name__ == "BatchNorm" for layer in control_norms)
    assert all(type(layer).__name__ == "GroupNorm" for layer in candidate_norms)

    control_keys = list(control.network.state_dict())
    candidate_keys = list(candidate.network.state_dict())
    control_running = sum(
        "running_mean" in key or "running_var" in key for key in control_keys
    )
    candidate_running = sum(
        "running_mean" in key or "running_var" in key for key in candidate_keys
    )
    assert control_running == 98
    assert candidate_running == 0

    candidate.train()
    x_np = np.random.RandomState(7).randn(1, 32, 3).astype(np.float32) * 0.02
    before = candidate_norms[0].weight.numpy().copy()
    optimizer = jt.optim.Adam(candidate.network.parameters(), lr=1e-4)
    output = candidate.network(jt.array(x_np))[0]
    loss = (output ** 2).mean()
    optimizer.step(loss)
    after = candidate_norms[0].weight.numpy().copy()
    update_max_abs = float(np.max(np.abs(after - before)))
    assert np.isfinite(output.numpy()).all()
    assert np.isfinite(float(loss.item()))
    assert update_max_abs > 0.0

    print(json.dumps({
        "wrapper_path": "PDLTSLight -> PDLTSLightNetwork",
        "norm_layer_count": len(candidate_norms),
        "control_norm_type": type(control_norms[0]).__name__,
        "candidate_norm_type": type(candidate_norms[0]).__name__,
        "control_running_stat_key_count": control_running,
        "candidate_running_stat_key_count": candidate_running,
        "candidate_state_key_count": len(candidate_keys),
        "optimizer_update_max_abs": update_max_abs,
        "finite_forward": True,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
