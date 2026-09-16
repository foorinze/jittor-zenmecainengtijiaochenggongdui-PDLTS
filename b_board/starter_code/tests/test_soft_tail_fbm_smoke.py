"""learnable soft-tail FBM 冒烟测试。"""

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
    hard_default = PDLTSLightNetwork(**kwargs)
    jt.set_global_seed(123)
    hard_explicit = PDLTSLightNetwork(
        fbm_mask_mode="hard", fbm_soft_tail_init=0.05, **kwargs
    )
    keys_default = list(hard_default.state_dict().keys())
    keys_explicit = list(hard_explicit.state_dict().keys())
    assert keys_default == keys_explicit
    hard_default.eval()
    hard_explicit.eval()
    out_default = hard_default(jt.array(x_np))[0]
    out_explicit = hard_explicit(jt.array(x_np))[0]
    hard_max_abs = float(jt.abs(out_default - out_explicit).max().item())
    # GPU reduction/order can introduce one-ULP noise across two independent
    # forward calls; the default and explicit-hard paths must remain equivalent.
    assert hard_max_abs <= 1e-7, hard_max_abs

    jt.set_global_seed(123)
    soft = PDLTSLightNetwork(
        fbm_mask_mode="learnable_soft_tail", fbm_soft_tail_init=0.05, **kwargs
    )
    soft.train()
    named = dict(soft.named_parameters())
    assert "fbm_tail_logit" in named
    assert tuple(named["fbm_tail_logit"].shape) == (1, 1, 24)
    trainable_named = collect_trainable_named_params(soft)
    trainable_names = [name for name, _ in trainable_named]
    assert "fbm_tail_logit" in trainable_names
    keep0 = jt.sigmoid(soft.fbm_tail_logit)
    assert abs(float(keep0.mean().item()) - 0.05) < 1e-6

    target = jt.array(x_np * 0.8)
    # AffineCoupling 的输出层为 zero-init；真实训练第 0 步先激活主干混合，
    # 此时 tail 对坐标输出尚无梯度。先按真实 optimizer 语义更新全体参数一步，
    # 再验证第二步 tail 梯度和更新，避免把预期初始化暂态误判为断图。
    warm_optimizer = jt.optim.Adam([param for _, param in trainable_named], lr=1e-3)
    warm_out = soft(jt.array(x_np))[0]
    warm_loss = ((warm_out - target) ** 2).mean()
    warm_optimizer.step(warm_loss)

    out = soft(jt.array(x_np))[0]
    loss = ((out - target) ** 2).mean()
    grad = jt.grad(loss, [soft.fbm_tail_logit])[0]
    grad_abs_sum = float(jt.abs(grad).sum().item())
    assert np.isfinite(float(loss.item())) and grad_abs_sum > 0.0

    before = soft.fbm_tail_logit.numpy().copy()
    optimizer = jt.optim.Adam([soft.fbm_tail_logit], lr=1e-2)
    out_step = soft(jt.array(x_np))[0]
    loss_step = ((out_step - target) ** 2).mean()
    optimizer.step(loss_step)
    change = float(np.abs(soft.fbm_tail_logit.numpy() - before).max())
    assert change > 0.0

    print(json.dumps({
        "hard_state_keys": len(keys_default),
        "hard_default_explicit_max_abs": hard_max_abs,
        "soft_extra_state_keys": len(soft.state_dict()) - len(keys_default),
        "soft_tail_parameter_count": int(soft.fbm_tail_logit.numel()),
        "tail_in_optimizer": True,
        "forward_loss": float(loss.item()),
        "tail_grad_abs_sum": grad_abs_sum,
        "adam_max_abs_change": change,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
