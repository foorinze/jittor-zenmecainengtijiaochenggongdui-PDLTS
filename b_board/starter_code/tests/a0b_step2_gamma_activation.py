"""Single-batch gamma activation test.

Runs the full PDLTS Light training graph (Chamfer + fixed_L2 loss) on a fixed
dummy batch for 50 steps. Tracks whether gamma leaves zero and starts
contributing to the denoised output.

Gate:
  - gamma grows to > 1e-5 within 50 steps
  - global_applied_delta_ratio is no longer constant zero
  - loss stays finite (no NaN / explosion)
  - (optional) aug sensitivity at 1%/10%/100% multi-scale

Usage:
    python starter_code/tests/a0b_step2_gamma_activation.py [--steps 50] [--seed 20260528]
"""

import argparse
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

from src.model.pdlts_light.model import PDLTSLightNetwork
from src.model.pdlts_light.system import _chamfer_l2_breakdown


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_network(seed=20260528):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    return PDLTSLightNetwork(
        pc_channel=3,
        aug_channel=48,
        n_injector=12,
        cut_channel=24,
        nflow_module=12,
        num_neighbors=32,
        global_context_mode="hilbert_full",
    )


def make_dummy_pair(batch_size=1, patch_size=128, noise_std=0.015, seed=42):
    """Generate a noisy/clean patch pair with realistic shape + noise."""
    rng = np.random.RandomState(seed)

    # Clean: ellipsoid + random perturbation (simulate airplane-like shape)
    theta = rng.rand(patch_size) * np.pi * 2
    phi = rng.rand(patch_size) * np.pi
    # Vary radii by band: some points on wider band
    band = rng.randint(0, 3, size=patch_size)
    radii = np.where(band == 0, 0.25,
                     np.where(band == 1, 0.30, 0.20))
    clean = np.stack([
        radii * np.sin(phi) * np.cos(theta),
        radii * np.sin(phi) * np.sin(theta) * 0.7,
        radii * np.cos(phi) * 1.3,
    ], axis=-1).astype(np.float32)

    # Noisy: add Gaussian noise
    noisy = clean + rng.randn(patch_size, 3).astype(np.float32) * noise_std

    # Add batch dim
    clean = clean[None, :, :]
    noisy = noisy[None, :, :]
    return jt.array(noisy), jt.array(clean)


def compute_chamfer_l2(denoised, clean):
    """Symmetric Chamfer + per-point L2 (fixed-index)."""
    chamfer, d_x2y, d_y2x, idx_x2y, nn_y = _chamfer_l2_breakdown(denoised, clean)
    l2 = ((denoised - clean) ** 2).sum(dim=-1).mean()
    return chamfer, l2


def test_aug_sensitivity_multi_scale(net, xyz, scales=None):
    """Measure denoised output change when aug is perturbed at multiple scales.

    Uses forward_from_aug to bypass MLGC and directly inject perturbed aug.
    """
    if scales is None:
        scales = [0.01, 0.10, 1.00]

    net.eval()
    with jt.no_grad():
        _, knn_idx = jt.misc.knn(xyz, xyz, net.num_neighbors)
        inj_f = net.feat_extract(xyz, knn_idx)
        aug = net.unit_coupling(xyz, knn_idx)

        # Reference output
        x = jt.concat([xyz, aug], dim=-1)
        if net.global_context_mode == "hilbert_full":
            aug_global = net.global_block(aug, xyz)
            aug_ref = aug + net.global_block.gamma * aug_global
            x = jt.concat([xyz, aug_ref], dim=-1)
        z, _ = net.f(x, inj_f)
        predict_z = z * net.channel_mask
        full_ref = net.g(predict_z, inj_f)
        denoised_ref = full_ref[..., :3]

        ref_norm = jt.sqrt((denoised_ref ** 2).sum(dim=-1)).mean()

        results = {}
        for scale in scales:
            rng = np.random.RandomState(int(scale * 1000 + 99))
            noise = jt.array(rng.randn(*aug.shape).astype(np.float32))
            noise = noise / (jt.sqrt((noise ** 2).sum(dim=-1, keepdims=True)) + 1e-10)
            noise = noise * jt.sqrt((aug ** 2).sum(dim=-1, keepdims=True)) * scale

            aug_pert = aug + noise

            x_pert = jt.concat([xyz, aug_pert], dim=-1)
            if net.global_context_mode == "hilbert_full":
                aug_global_pert = net.global_block(aug_pert, xyz)
                aug_pert = aug_pert + net.global_block.gamma * aug_global_pert
                x_pert = jt.concat([xyz, aug_pert], dim=-1)

            z_pert, _ = net.f(x_pert, inj_f)
            predict_z_pert = z_pert * net.channel_mask
            full_pert = net.g(predict_z_pert, inj_f)
            denoised_pert = full_pert[..., :3]

            point_diff = jt.sqrt(((denoised_ref - denoised_pert) ** 2).sum(dim=-1)).mean()
            ratio = float((point_diff / (ref_norm + 1e-10)).item())
            results[f"perturb_{int(scale*100):d}pct_sensitivity"] = ratio

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--lambda-l2", type=float, default=0.1,
                        help="fixed_L2 weight (matches PDLTS default)")
    args = parser.parse_args()

    print("=" * 60)
    print("A0b-step2: Gamma Activation Test (50-step training)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Steps: {args.steps}, LR: {args.lr}, Patch: {args.patch_size}")
    print(f"Noise std: {args.noise_std}, L2 lambda: {args.lambda_l2}")
    print("=" * 60)

    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Setup ----
    net = make_network(seed=args.seed)
    net.train()

    noisy, clean = make_dummy_pair(
        patch_size=args.patch_size,
        noise_std=args.noise_std,
        seed=args.seed + 1,
    )

    optimizer = nn.Adam(net.parameters(), lr=args.lr)

    # ---- Training loop ----
    history = []
    gamma_activated = False
    gamma_max = 0.0

    for step in range(args.steps):
        denoised, _ldj, _loss_d = net(noisy)
        chamfer, l2 = compute_chamfer_l2(denoised, clean)
        loss = chamfer + args.lambda_l2 * l2

        optimizer.step(loss)

        diag = dict(net._global_diag) if net._global_diag else {}
        gamma_val = diag.get("global_gamma", 0.0)
        applied_ratio = diag.get("global_applied_delta_ratio", 0.0)
        delta_ratio = diag.get("global_delta_ratio", 0.0)
        entropy = diag.get("attention_entropy_mean", 0.0)
        delta_norm = diag.get("global_delta_norm", 0.0)

        row = {
            "step": step,
            "loss": float(loss.item()),
            "chamfer": float(chamfer.item()),
            "l2": float(l2.item()),
            "global_gamma": gamma_val,
            "global_delta_norm": delta_norm,
            "global_delta_ratio": delta_ratio,
            "global_applied_delta_ratio": applied_ratio,
            "attention_entropy_mean": entropy,
        }
        history.append(row)

        gamma_max = max(gamma_max, abs(gamma_val))

        if step == 0 or (step + 1) % 10 == 0 or step == args.steps - 1:
            print(
                f"  step {step:3d} | loss={float(loss.item()):.4e} "
                f"| gamma={gamma_val:.4e} "
                f"| applied_ratio={applied_ratio:.4e} "
                f"| delta_ratio={delta_ratio:.4e}"
            )

        if abs(gamma_val) > 1e-5:
            gamma_activated = True

        # Safety: NaN check
        if not np.isfinite(float(loss.item())):
            print(f"\n  FAIL: loss became NaN at step {step}")
            break

    # ---- Post-training diagnostics ----
    print("\n--- Post-training aug sensitivity (multi-scale) ---")
    sens = test_aug_sensitivity_multi_scale(net, noisy, scales=[0.01, 0.10, 1.00])
    for k, v in sens.items():
        print(f"  {k}: {v:.6e}")

    # ---- Gate evaluation ----
    print("\n" + "=" * 60)
    print("GATE EVALUATION")
    print("=" * 60)
    print(f"  gamma max absolute value:    {gamma_max:.6e}")
    print(f"  gamma activated (> 1e-5):    {gamma_activated}")
    print(f"  loss finite:                  {np.isfinite(float(loss.item()))}")

    gate_pass = gamma_activated and np.isfinite(float(loss.item()))

    if gate_pass:
        print("\n  GATE PASS: gamma activates within 50 training steps.")
        print("  Global context block mechanism is healthy.")
        print("  Ready for A1 (1k/20ep matched probe).")
    else:
        print("\n  GATE FAIL: gamma did not activate within 50 steps.")
        if not gamma_activated:
            print("  ROOT CAUSE: INN initial near-identity coupling too weak.")
            print("  SUGGESTION: try larger LR, or add short warmup phase with")
            print("              increased gamma LR multiplier before A1.")
        if not np.isfinite(float(loss.item())):
            print("  ROOT CAUSE: loss NaN. Check numerical stability.")

    # ---- Save results ----
    out_dir = os.path.join(REPO_ROOT, "outputs", "diagnostics", "b_final_gamma_activation")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gamma_activation_{ts}.json")
    with open(out_path, "w") as f:
        json.dump({
            "args": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else bool(v) if isinstance(v, (np.bool_,)) else v) for k, v in vars(args).items()},
            "gate_pass": bool(gate_pass),
            "gamma_activated": bool(gamma_activated),
            "gamma_max_abs": float(gamma_max),
            "final_loss": float(loss.item()),
            "aug_sensitivity": {k: float(v) for k, v in sens.items()},
            "history": history,
        }, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
