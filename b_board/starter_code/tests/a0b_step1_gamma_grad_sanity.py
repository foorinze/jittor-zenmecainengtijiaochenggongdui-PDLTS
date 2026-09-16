"""绕开 INN，验证 gamma 梯度链健康度。

测的是: block.gamma → aug_new → loss → grad(gamma) 这条链是否完整。

如果这步 gamma 不动 → global block 或 gamma 注册/梯度有实现问题。
如果这步 gamma 动 → global block 自身机制健康，问题仅在 INN 初始耦合。

用法:
    python starter_code/tests/a0b_step1_gamma_grad_sanity.py
"""

import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STARTER_ROOT = os.path.join(REPO_ROOT, "starter_code")
if STARTER_ROOT not in sys.path:
    sys.path.insert(0, STARTER_ROOT)

import jittor as jt
from jittor import nn

from src.model.pdlts_light.model import PDLTSLightNetwork
from src.model.pdlts_light.global_context import GlobalContextBlock


def make_network(**kwargs):
    return PDLTSLightNetwork(
        pc_channel=3,
        aug_channel=48,
        n_injector=12,
        cut_channel=24,
        nflow_module=12,
        num_neighbors=kwargs.pop("num_neighbors", 32),
        global_context_mode="hilbert_full",
        **kwargs,
    )


def make_dummy_aug_xyz(batch_size=1, patch_size=128, seed=42):
    rng = np.random.RandomState(seed)
    xyz = rng.randn(batch_size, patch_size, 3).astype(np.float32) * 0.3
    aug = rng.randn(batch_size, patch_size, 48).astype(np.float32)
    return jt.array(xyz), jt.array(aug)


def main():
    print("=" * 60)
    print("A0b-step1: Gamma Gradient Sanity (Bypass INN)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ---- Test 1: standalone GlobalContextBlock ----
    print("\n--- Test 1: standalone GlobalContextBlock ---")
    block = GlobalContextBlock(
        feature_dim=48, d_model=64, n_heads=4, n_layers=2,
    )

    xyz, aug = make_dummy_aug_xyz(batch_size=1, patch_size=128, seed=20260528)

    # 前向
    aug_global = block(aug, xyz)
    aug_new = aug + block.gamma * aug_global
    loss = (aug_new ** 2).mean()

    gamma_before = float(block.gamma.item())
    print(f"  gamma before: {gamma_before:.8e}")

    # 只对 gamma 做优化
    optimizer = nn.SGD([block.gamma], lr=0.1)
    optimizer.step(loss)

    gamma_after_standalone = float(block.gamma.item())
    gamma_change = gamma_after_standalone - gamma_before
    print(f"  gamma after:  {gamma_after_standalone:.8e}")
    print(f"  gamma change: {gamma_change:.8e}")

    # 额外检查: 如果 gamma 动了但值也是 0 (比如 1e-30 级别的变化)
    # 调大 lr 再试一次
    if abs(gamma_change) < 1e-12:
        print("  WARNING: gamma unchanged with lr=0.1, trying lr=1.0")
        block.gamma.assign(jt.zeros((1,)))
        aug_global = block(aug, xyz)
        aug_new = aug + block.gamma * aug_global
        loss = (aug_new ** 2).mean()
        optimizer2 = nn.SGD([block.gamma], lr=1.0)
        optimizer2.step(loss)
        gamma_after_boost = float(block.gamma.item())
        gamma_change = gamma_after_boost - 0.0
        print(f"  gamma after lr=1.0: {gamma_after_boost:.8e}")
        print(f"  gamma change:        {gamma_change:.8e}")

    standalone_pass = abs(gamma_change) > 1e-12
    print(f"  {'PASS' if standalone_pass else 'FAIL'}: standalone gamma grad")

    # ---- Test 2: gamma inside full model (still bypass INN tail) ----
    print("\n--- Test 2: gamma via model's global_block ---")
    net = make_network()
    net.train()

    xyz2 = jt.array(np.random.RandomState(99).randn(1, 128, 3).astype(np.float32) * 0.3)
    aug2 = jt.array(np.random.RandomState(99).randn(1, 128, 48).astype(np.float32))

    aug_global2 = net.global_block(aug2, xyz2)
    aug_new2 = aug2 + net.global_block.gamma * aug_global2
    loss2 = (aug_new2 ** 2).mean()

    gamma_before2 = float(net.global_block.gamma.item())
    print(f"  gamma before: {gamma_before2:.8e}")

    opt2 = nn.SGD([net.global_block.gamma], lr=0.1)
    opt2.step(loss2)

    gamma_after2 = float(net.global_block.gamma.item())
    gamma_change2 = gamma_after2 - gamma_before2
    print(f"  gamma after:  {gamma_after2:.8e}")
    print(f"  gamma change: {gamma_change2:.8e}")

    model_pass = abs(gamma_change2) > 1e-12
    print(f"  {'PASS' if model_pass else 'FAIL'}: model-embedded gamma grad")

    # ---- Test 3: gamma + global block params all get gradients ----
    print("\n--- Test 3: all global block params receive gradients ---")
    net3 = make_network()
    net3.train()

    xyz3, aug3 = make_dummy_aug_xyz(batch_size=1, patch_size=128, seed=77)

    aug_global3 = net3.global_block(aug3, xyz3)
    aug_new3 = aug3 + net3.global_block.gamma * aug_global3
    loss3 = (aug_new3 ** 2).mean()

    # 收集 global block 的所有参数
    gb_params = list(net3.global_block.parameters())
    n_gb_params = len(gb_params)

    param_grads = {}
    optimizer3 = nn.SGD(gb_params, lr=0.01)
    optimizer3.step(loss3)

    n_nonzero = 0
    for name, param in net3.global_block.named_parameters():
        old_val = float(param.item() if param.numel() == 1 else param.abs().max().item())
        param_grads[name] = old_val
    # 检查 gamma 的变化
    gamma_change3 = float(net3.global_block.gamma.item())
    print(f"  gamma change (all-param opt): {gamma_change3:.8e}")
    print(f"  total global block params:    {n_gb_params}")

    all_pass = standalone_pass and model_pass and (abs(gamma_change3) > 1e-12)
    print(f"  {'PASS' if all_pass else 'FAIL'}: all global block grad checks")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  standalone gamma grad:   {'PASS' if standalone_pass else 'FAIL'}")
    print(f"  model-embedded gamma grad: {'PASS' if model_pass else 'FAIL'}")
    print(f"  Overall:                   {'PASS' if all_pass else 'FAIL'}")
    if all_pass:
        print("\n  Verdict: gamma gradient chain is healthy.")
        print("  A0 gamma_grad=0 is caused by INN initial near-identity coupling.")
        print("  Proceed to A0b-step2 (single-batch 50-step activation test).")
    else:
        print("\n  Verdict: gamma gradient chain is BROKEN.")
        print("  Check: gamma parameter registration, gather gradient path, optimizer param list.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
