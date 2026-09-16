"""AffineCoupling log-scale 增益冒烟测试。"""

from __future__ import annotations

import json

import jittor as jt
import numpy as np

from src.model.pdlts_light.model import PDLTSLightNetwork


def coupling_clamps(network: PDLTSLightNetwork) -> list[float]:
    return [
        float(layer.log_scale_clamp)
        for assembly in network.flow_assemblies
        for layer in assembly.chain
        if hasattr(layer, "log_scale_clamp")
    ]


def main() -> None:
    jt.flags.use_cuda = 1
    kwargs = dict(num_neighbors=8, mlgc_hidden=32, coupling_hidden=32)
    x_np = np.random.RandomState(7).randn(1, 32, 3).astype(np.float32) * 0.02

    jt.set_global_seed(123)
    control = PDLTSLightNetwork(log_scale_clamp=0.1, **kwargs)
    jt.set_global_seed(123)
    candidate = PDLTSLightNetwork(log_scale_clamp=0.2, **kwargs)

    control_clamps = coupling_clamps(control)
    candidate_clamps = coupling_clamps(candidate)
    assert len(control_clamps) == len(candidate_clamps) == 24
    assert set(control_clamps) == {0.1}
    assert set(candidate_clamps) == {0.2}
    assert list(control.state_dict().keys()) == list(candidate.state_dict().keys())
    assert sum(param.numel() for param in control.parameters()) == sum(
        param.numel() for param in candidate.parameters()
    )

    control.eval()
    candidate.eval()
    control_out = control(jt.array(x_np))[0]
    candidate_out = candidate(jt.array(x_np))[0]
    initial_forward_max_abs = float(jt.abs(control_out - candidate_out).max().item())
    # Coupling 输出层为 zero-init，clamp 改变不能扰动初始前向。
    assert initial_forward_max_abs <= 1e-7, initial_forward_max_abs

    print(json.dumps({
        "coupling_count": len(control_clamps),
        "control_clamp": control_clamps[0],
        "candidate_clamp": candidate_clamps[0],
        "state_keys_equal": True,
        "parameter_count_equal": True,
        "initial_forward_max_abs": initial_forward_max_abs,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
