"""Global context block smoke test.

This script checks implementation safety only. It does not start training and
does not claim that the global-context mechanism improves CD.

Run:
    cd <repo_root>
    python starter_code/tests/a0_global_context_smoke.py
"""

import json
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

from src.model.pdlts_light.hilbert import hilbert_encode, hilbert_sort_indices
from src.model.pdlts_light.model import PDLTSLightNetwork


def make_dummy_patch(batch_size=1, patch_size=64, seed=42):
    rng = np.random.RandomState(seed)
    xyz = rng.randn(batch_size, patch_size, 3).astype(np.float32) * 0.3
    theta = rng.rand(patch_size) * np.pi * 2
    phi = rng.rand(patch_size) * np.pi
    radii = rng.rand(patch_size) * 0.3 + 0.1
    ellipsoid = np.stack(
        [
            radii * np.sin(phi) * np.cos(theta),
            radii * np.sin(phi) * np.sin(theta) * 0.7,
            radii * np.cos(phi) * 1.3,
        ],
        axis=-1,
    ).astype(np.float32)
    return xyz * 0.5 + ellipsoid[None, :, :] * 0.5


def make_network(global_context_mode="off", seed=42, **kwargs):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    return PDLTSLightNetwork(
        pc_channel=3,
        aug_channel=48,
        n_injector=12,
        cut_channel=24,
        nflow_module=12,
        num_neighbors=kwargs.pop("num_neighbors", 32),
        global_context_mode=global_context_mode,
        **kwargs,
    )


def compute_hilbert_quality(coords_np, bits=10):
    B, N, D = coords_np.shape
    sort_idx, _ = hilbert_sort_indices(coords_np, bits=bits)
    results = {}
    for b in range(B):
        c_int = np.clip(
            np.floor(
                (coords_np[b] - coords_np[b].min(axis=0))
                / (coords_np[b].max(axis=0) - coords_np[b].min(axis=0) + 1e-10)
                * ((1 << bits) - 1)
            ).astype(np.int64),
            0,
            (1 << bits) - 1,
        )
        codes = hilbert_encode(c_int, num_dims=D, num_bits=bits)
        unique_ratio = len(np.unique(codes)) / N

        si = sort_idx[b]
        hilbert_adj = coords_np[b, si[1:]] - coords_np[b, si[:-1]]
        hilbert_adj_dist = np.sqrt((hilbert_adj**2).sum(axis=-1)).mean()

        rand_order = np.random.RandomState(42).permutation(N)
        rand_adj = coords_np[b, rand_order[1:]] - coords_np[b, rand_order[:-1]]
        rand_adj_dist = np.sqrt((rand_adj**2).sum(axis=-1)).mean()

        results[f"batch_{b}_unique_ratio"] = float(unique_ratio)
        results[f"batch_{b}_hilbert_adj_dist"] = float(hilbert_adj_dist)
        results[f"batch_{b}_random_adj_dist"] = float(rand_adj_dist)
        results[f"batch_{b}_locality_ratio"] = float(hilbert_adj_dist / (rand_adj_dist + 1e-10))
    return results


def shared_param_max_diff(base, candidate):
    cand_params = dict(candidate.named_parameters())
    max_diff = 0.0
    checked = 0
    for name, param in base.named_parameters():
        if name not in cand_params:
            continue
        diff = float((param - cand_params[name]).abs().max().item())
        max_diff = max(max_diff, diff)
        checked += 1
    return max_diff, checked


def copy_shared_params(base, candidate):
    """Copy all non-global parameters from base into candidate for identity checks."""
    cand_params = dict(candidate.named_parameters())
    copied = 0
    for name, param in base.named_parameters():
        if name not in cand_params:
            continue
        cand_params[name].assign(param.copy())
        copied += 1
    return copied


def forward_from_aug(net, xyz, inj_f, aug):
    x = jt.concat([xyz, aug], dim=-1)
    z, ldj = net.f(x, inj_f)
    predict_z = z * net.channel_mask
    full_x = net.g(predict_z, inj_f)
    return full_x[..., : net.pc_channel], ldj


def compute_position_delta_cos(block, aug, xyz):
    """Compare global block output with learned position embedding vs zeroed embedding."""
    with jt.no_grad():
        out_with_pos = block(aug, xyz)
        old_pos = block.pos_emb.weight.copy()
        block.pos_emb.weight.assign(jt.zeros_like(block.pos_emb.weight))
        out_no_pos = block(aug, xyz)
        block.pos_emb.weight.assign(old_pos)

        flat_a = out_with_pos.reshape(-1)
        flat_b = out_no_pos.reshape(-1)
        cos_sim = (
            (flat_a * flat_b).sum()
            / (jt.sqrt((flat_a**2).sum()) * jt.sqrt((flat_b**2).sum()) + 1e-10)
        )
    return float(cos_sim.item())


def test_gamma_zero_identity():
    print("\n=== Check 4: gamma=0 identity ===")
    base = make_network(global_context_mode="off", seed=20260528)
    net_a = make_network(global_context_mode="hilbert_full", seed=20260528)

    copied = copy_shared_params(base, net_a)
    shared_diff, checked = shared_param_max_diff(base, net_a)
    print(f"  shared params copied:  {copied}")
    print(f"  shared params checked: {checked}")
    print(f"  shared param max diff: {shared_diff:.2e}")
    assert shared_diff < 1e-7, "shared params differ; identity test is invalid"

    gamma_val = float(net_a.global_block.gamma.item())
    assert abs(gamma_val) < 1e-8, f"gamma should be 0 at init, got {gamma_val}"
    print(f"  gamma initial value: {gamma_val:.2e}")

    xyz = jt.array(make_dummy_patch(batch_size=1, patch_size=64, seed=42))
    base.eval()
    net_a.eval()
    with jt.no_grad():
        denoised_base, _, _ = base(xyz)
        denoised_a, _, _ = net_a(xyz)

    diff = (denoised_base - denoised_a).abs()
    max_diff = float(diff.max().item())
    mean_diff = float(diff.mean().item())
    print(f"  max  difference: {max_diff:.2e}")
    print(f"  mean difference: {mean_diff:.2e}")
    return max_diff < 1e-6, max_diff


def test_gamma_gradient():
    print("\n=== Check 5: grad(gamma) after backward ===")
    net = make_network(global_context_mode="hilbert_full", seed=20260528)
    net.train()
    xyz = jt.array(make_dummy_patch(batch_size=1, patch_size=64, seed=42))
    denoised, _, _ = net(xyz)
    loss = (denoised**2).mean()

    gamma_before = float(net.global_block.gamma.item())
    optimizer = nn.SGD(net.parameters(), lr=0.01)
    optimizer.step(loss)
    gamma_after = float(net.global_block.gamma.item())
    gamma_change = gamma_after - gamma_before

    print(f"  gamma before backward: {gamma_before:.6e}")
    print(f"  gamma after  backward: {gamma_after:.6e}")
    print(f"  gamma change:          {gamma_change:.6e}")
    return abs(gamma_change) > 1e-10, gamma_change


def test_perturbation_sensitivity():
    print("\n=== Check 6: aug feature perturbation sensitivity ===")
    net = make_network(global_context_mode="off", seed=20260528)
    net.eval()
    xyz = jt.array(make_dummy_patch(batch_size=1, patch_size=64, seed=42))

    with jt.no_grad():
        _, knn_idx = jt.misc.knn(xyz, xyz, net.num_neighbors)
        inj_f = net.feat_extract(xyz, knn_idx)
        aug = net.unit_coupling(xyz, knn_idx)
        denoised_ref, _ = forward_from_aug(net, xyz, inj_f, aug)

        rng = np.random.RandomState(99)
        noise = jt.array(rng.randn(*aug.shape).astype(np.float32))
        scale = jt.sqrt((aug**2).mean()) * 0.01
        denoised_pert, _ = forward_from_aug(net, xyz, inj_f, aug + noise * scale)

    point_diff = jt.sqrt(((denoised_ref - denoised_pert) ** 2).sum(dim=-1)).mean()
    point_norm = jt.sqrt((denoised_ref**2).sum(dim=-1)).mean()
    ratio = float((point_diff / (point_norm + 1e-10)).item())
    print(f"  movement / norm ratio: {ratio:.6f}")
    return ratio


def record_diagnostics():
    print("\n=== Check 7: diagnostics recording ===")
    net = make_network(global_context_mode="hilbert_full", seed=20260528)
    xyz = jt.array(make_dummy_patch(batch_size=1, patch_size=64, seed=42))
    with jt.no_grad():
        _ = net(xyz)
    diag = net._global_diag
    required = [
        "global_gamma",
        "global_delta_norm",
        "feature_mlgc_norm",
        "global_delta_ratio",
        "global_applied_delta_ratio",
        "attention_entropy_mean",
    ]
    for key in required:
        assert key in diag, f"missing diagnostic key: {key}"
        print(f"  {key}: {diag[key]}")
    return diag


def main():
    print("=" * 60)
    print("Global context block smoke test")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Jittor flags: use_cuda={jt.flags.use_cuda}")
    print("=" * 60)

    results = {}
    all_pass = True

    print("\n=== Check 1 & 2: Hilbert encoding quality ===")
    coords = make_dummy_patch(batch_size=2, patch_size=128, seed=123)
    hq = compute_hilbert_quality(coords, bits=10)
    results.update(hq)
    for k, v in sorted(hq.items()):
        print(f"  {k}: {v:.4f}")

    unique_ratios = [v for k, v in hq.items() if k.endswith("_unique_ratio")]
    min_unique = min(unique_ratios) if unique_ratios else 0.0
    if min_unique < 0.90:
        print("  FAIL: unique_ratio < 0.90")
        all_pass = False
    else:
        print("  PASS: unique_ratio >= 0.90")

    locality_ratios = [v for k, v in hq.items() if k.endswith("_locality_ratio")]
    avg_locality = float(np.mean(locality_ratios)) if locality_ratios else 1.0
    if avg_locality < 0.8:
        print("  PASS: Hilbert localizes better than random")
    else:
        print("  INFO: locality_ratio >= 0.8")
    results["avg_locality_ratio"] = avg_locality

    print("\n=== Check 3: position embedding effect ===")
    net_3 = make_network(global_context_mode="hilbert_full", seed=20260528)
    net_3.eval()
    xyz_3 = jt.array(make_dummy_patch(batch_size=1, patch_size=256, seed=456))
    with jt.no_grad():
        _, knn_idx = jt.misc.knn(xyz_3, xyz_3, net_3.num_neighbors)
        aug_3 = net_3.unit_coupling(xyz_3, knn_idx)
    pos_delta_cos = compute_position_delta_cos(net_3.global_block, aug_3, xyz_3)
    print(f"  position_delta_cos: {pos_delta_cos:.6f}")
    results["position_delta_cos"] = pos_delta_cos
    if pos_delta_cos >= 0.99:
        print("  INFO: position embedding effect is weak at init")
    else:
        print("  PASS: position embedding changes block output")

    identity_pass, max_diff = test_gamma_zero_identity()
    results["gamma_zero_max_diff"] = float(max_diff)
    all_pass = all_pass and identity_pass
    print("  PASS" if identity_pass else "  FAIL")

    grad_pass, gamma_change = test_gamma_gradient()
    results["gamma_grad_change"] = float(gamma_change)
    all_pass = all_pass and grad_pass
    print("  PASS" if grad_pass else "  FAIL")

    sens_ratio = test_perturbation_sensitivity()
    results["perturbation_sensitivity_ratio"] = float(sens_ratio)

    diag = record_diagnostics()
    results.update({k: float(v) for k, v in diag.items()})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  All critical checks pass: {all_pass}")
    print(f"  Ready for A1 (1k/20ep):  {all_pass}")

    out_dir = os.path.join(REPO_ROOT, "outputs", "diagnostics", "b_final_global_context_smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"a0_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results written to: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
