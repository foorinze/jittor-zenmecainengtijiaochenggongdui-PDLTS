"""固定预算 Top-k STE FBM 冒烟测试。"""

from __future__ import annotations

import json

import jittor as jt
import numpy as np

from src.model.pdlts_light.model import PDLTSLightNetwork
from src.system.pdlts_light import collect_trainable_named_params


def main() -> None:
    jt.flags.use_cuda = 1
    kwargs = dict(num_neighbors=8, mlgc_hidden=32, coupling_hidden=32)
    x_np = np.random.RandomState(7).randn(1, 32, 3).astype(np.float32) * 0.02

    jt.set_global_seed(123)
    hard = PDLTSLightNetwork(fbm_mask_mode="hard", **kwargs)
    jt.set_global_seed(123)
    topkste = PDLTSLightNetwork(
        fbm_mask_mode="learnable_topk_ste",
        fbm_topk_init_margin=0.5,
        **kwargs,
    )
    named = dict(topkste.named_parameters())
    assert "fbm_mask_logit" in named
    assert tuple(named["fbm_mask_logit"].shape) == (1, 1, 51)
    trainable_names = [name for name, _ in collect_trainable_named_params(topkste)]
    assert "fbm_mask_logit" in trainable_names

    ste_mask, hard_mask = topkste._fbm_topk_ste_mask()
    hard_mask_np = hard_mask.numpy()
    assert int(hard_mask_np.sum()) == 27
    assert np.array_equal(hard_mask_np, topkste.channel_mask.numpy())
    assert float(np.abs(ste_mask.numpy() - hard_mask_np).max()) <= 1e-7

    hard.eval()
    topkste.eval()
    out_hard = hard(jt.array(x_np))[0]
    out_topkste = topkste(jt.array(x_np))[0]
    initial_forward_max_abs = float(jt.abs(out_hard - out_topkste).max().item())
    assert initial_forward_max_abs <= 1e-7, initial_forward_max_abs

    # 人工交换边界两侧通道，验证预算固定且通道身份可改变。
    swap_logits = np.full((1, 1, 51), -0.5, dtype=np.float32)
    swap_logits[..., :27] = 0.5
    swap_logits[..., 26] = -1.0
    swap_logits[..., 27] = 1.0
    topkste.fbm_mask_logit.assign(jt.array(swap_logits))
    _, swapped_hard = topkste._fbm_topk_ste_mask()
    swapped_np = swapped_hard.numpy().reshape(-1)
    assert int(swapped_np.sum()) == 27
    assert swapped_np[26] == 0.0 and swapped_np[27] == 1.0

    topkste.train()
    trainable_named = collect_trainable_named_params(topkste)
    target = jt.array(x_np * 0.8)
    # AffineCoupling 输出层为 zero-init；先更新主干一步，再测掩码梯度。
    warm_optimizer = jt.optim.Adam([param for _, param in trainable_named], lr=1e-3)
    warm_out = topkste(jt.array(x_np))[0]
    warm_loss = ((warm_out - target) ** 2).mean()
    warm_optimizer.step(warm_loss)

    out = topkste(jt.array(x_np))[0]
    loss = ((out - target) ** 2).mean()
    grad = jt.grad(loss, [topkste.fbm_mask_logit])[0]
    grad_abs_sum = float(jt.abs(grad).sum().item())
    assert np.isfinite(float(loss.item())) and grad_abs_sum > 0.0

    before = topkste.fbm_mask_logit.numpy().copy()
    optimizer = jt.optim.Adam([topkste.fbm_mask_logit], lr=1e-2)
    out_step = topkste(jt.array(x_np))[0]
    loss_step = ((out_step - target) ** 2).mean()
    optimizer.step(loss_step)
    change = float(np.abs(topkste.fbm_mask_logit.numpy() - before).max())
    assert change > 0.0

    print(json.dumps({
        "hard_state_keys": len(hard.state_dict()),
        "topkste_extra_state_keys": len(topkste.state_dict()) - len(hard.state_dict()),
        "topkste_parameter_count": int(topkste.fbm_mask_logit.numel()),
        "initial_keep_count": int(hard_mask_np.sum()),
        "initial_forward_max_abs": initial_forward_max_abs,
        "forced_swap_out": 26,
        "forced_swap_in": 27,
        "forward_loss": float(loss.item()),
        "mask_grad_abs_sum": grad_abs_sum,
        "adam_max_abs_change": change,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
