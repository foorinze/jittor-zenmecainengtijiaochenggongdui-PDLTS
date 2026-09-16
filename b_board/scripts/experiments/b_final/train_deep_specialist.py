#!/usr/bin/env python3
"""B 榜最终方案第二遍 specialist（专训模型）训练。

最终 81.05 使用第二遍专训：warm-start B 榜 ep149 base，在固定 2000 样本上训练
50 epochs / lr 1e-4 / batch 16。每个 shape 每个 epoch 重采 20 个 1024 点
patch，采用 FPS seed 点中心化，seed=42，EMA 不启用。

用法：
    python scripts/experiments/b_final/train_deep_specialist.py \\
        --stage specialist \\
        --train-datalist starter_code/datalist/b_final_specialist_train2k.txt \\
        --firstpass-cache <ep149 对 2000 样本的 merged pred/ 目录> \\
        --load-ckpt <ep149 checkpoint>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT / "starter_code"))

import jittor as jt  # noqa: E402

jt.flags.use_cuda = 1

VERSION = "b_final"
SCRIPT_VERSION = "public"

DEFAULT_TRAIN_ROOT = "dataset/train_b"
STAGE_DEFAULTS = {
    "specialist": {"epochs": 50, "lr": 1e-4, "batch_size": 16},
}
PATCH_SIZE = 1024
N_PATCHES_PER_SAMPLE = 20
SEED = 42


def read_datalist(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8")
            if line.strip() and not line.startswith("#")]


def rel_path(root: Path, path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def fps_indices(pts: np.ndarray, n_seeds: int, rng: np.random.RandomState) -> np.ndarray:
    """最远点采样（FPS），沿用最终方案的 specialist（专训模型）训练契约。"""
    n_pts = len(pts)
    if n_seeds >= n_pts:
        return np.arange(n_pts)
    selected = np.zeros(n_seeds, dtype=np.int64)
    dists = np.full(n_pts, np.inf)
    selected[0] = rng.randint(n_pts)
    for i in range(1, n_seeds):
        d = np.sum((pts - pts[selected[i - 1]]) ** 2, axis=1)
        dists = np.minimum(dists, d)
        selected[i] = np.argmax(dists)
    return selected


def extract_patches_one_shape(
    input_pts: np.ndarray,
    clean_pts: np.ndarray,
    n_patches: int,
    patch_size: int,
    patch_rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """最终 patch（局部块）契约：FPS seed 点中心化（不是 patch 均值）。"""
    from scipy.spatial import cKDTree

    seeds = fps_indices(input_pts, n_patches, patch_rng)
    tree = cKDTree(input_pts)
    patches_in, patches_cl = [], []
    for seed_idx in seeds:
        _, idx = tree.query(input_pts[seed_idx], k=patch_size)
        patch_in = input_pts[idx]
        patch_cl = clean_pts[idx]
        center = input_pts[seed_idx]  # 最终契约：seed 点中心化
        patches_in.append((patch_in - center).astype(np.float32))
        patches_cl.append((patch_cl - center).astype(np.float32))
    return np.array(patches_in), np.array(patches_cl), seeds


def extract_epoch_patches(
    rels: list[str],
    firstpass_root: Path,
    clean_root: Path,
    n_patches_per_sample: int,
    patch_size: int,
    patch_rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, str]:
    all_in, all_cl, all_seeds = [], [], []
    for rel in rels:
        fp = np.load(firstpass_root / rel / "denoised.npy").astype(np.float32)
        cl = np.load(clean_root / rel / "clean.npy").astype(np.float32)
        pi, pc, seeds = extract_patches_one_shape(
            fp, cl, n_patches_per_sample, patch_size, patch_rng)
        all_in.append(pi)
        all_cl.append(pc)
        all_seeds.append(seeds)
    all_in = np.concatenate(all_in, axis=0)
    all_cl = np.concatenate(all_cl, axis=0)
    concat_seeds = np.concatenate(all_seeds).astype(np.int64)
    patch_index_hash = hashlib.sha256(concat_seeds.tobytes()).hexdigest()
    return all_in, all_cl, patch_index_hash


def load_model(ckpt_path: Path, mlgc_hidden: int = 64):
    from src.model.pdlts_light.model import PDLTSLightNetwork

    model = PDLTSLightNetwork(mlgc_hidden=mlgc_hidden)
    state = jt.load(str(ckpt_path))
    if any(k.startswith("network.") for k in state):
        state = {k.replace("network.", ""): v for k, v in state.items()
                 if k.startswith("network.")}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatch = sorted(
        key for key in set(expected) & set(state)
        if tuple(expected[key].shape) != tuple(state[key].shape)
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "checkpoint/model contract mismatch: "
            f"mlgc_hidden={mlgc_hidden}, missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}, shape_mismatch={shape_mismatch[:8]}"
        )
    model.load_state_dict(state)
    print(
        f"  checkpoint contract: {len(expected)} keys, "
        f"mlgc_hidden={mlgc_hidden}, missing=0, unexpected=0, shape_mismatch=0"
    )
    return model


def chamfer_l2(pred: jt.Var, target: jt.Var) -> jt.Var:
    _, idx_p2t = jt.misc.knn(pred, target, 1)
    batch, n_pred, _ = pred.shape
    _, n_target, _ = target.shape
    bi = jt.arange(batch).view(-1, 1)
    nn_t = target[bi, idx_p2t.reshape(batch, n_pred)]
    d_p2t = ((pred - nn_t) ** 2).sum(dim=-1).mean()

    _, idx_t2p = jt.misc.knn(target, pred, 1)
    nn_p = pred[bi, idx_t2p.reshape(batch, n_target)]
    d_t2p = ((target - nn_p) ** 2).sum(dim=-1).mean()
    return d_p2t + d_t2p


def train_c_contract(
    model,
    rels: list[str],
    firstpass_root: Path,
    clean_root: Path,
    n_epochs: int,
    n_patches_per_sample: int,
    patch_size: int,
    lr: float,
    batch_size: int,
    patch_seed: int,
    shuffle_seed: int,
):
    """最终训练契约：每 epoch 重采 + FPS seed 点中心化。"""
    optimizer = jt.optim.Adam(model.parameters(), lr=lr)
    patch_rng = np.random.RandomState(patch_seed)
    shuffle_rng = np.random.RandomState(shuffle_seed)

    epoch_records = []
    history = []
    global_step = 0

    for ep in range(n_epochs):
        model.train()
        all_in, all_cl, patch_index_hash = extract_epoch_patches(
            rels, firstpass_root, clean_root, n_patches_per_sample, patch_size, patch_rng)

        n_total = len(all_in)
        perm = shuffle_rng.permutation(n_total)
        batch_order_hash = hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_total, batch_size):
            idx = perm[start:start + batch_size]
            inp = jt.array(all_in[idx])
            tgt = jt.array(all_cl[idx])
            denoised, _, _ = model(inp)
            loss = chamfer_l2(denoised, tgt)
            optimizer.step(loss)
            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            if n_batches % 300 == 0:
                print(f"    step {global_step} (ep batch {n_batches}/"
                      f"{(n_total + batch_size - 1) // batch_size}) "
                      f"running_loss={epoch_loss / n_batches:.8f}", flush=True)

        avg_loss = epoch_loss / max(n_batches, 1)
        history.append(avg_loss)
        epoch_records.append({
            "epoch": ep + 1,
            "n_patches": int(n_total),
            "n_batches": n_batches,
            "loss": avg_loss,
            "patch_index_hash": patch_index_hash,
            "batch_order_hash": batch_order_hash,
        })
        print(f"  ep{ep + 1:3d}/{n_epochs}: loss={avg_loss:.8f} "
              f"patch_hash={patch_index_hash[:12]} batch_hash={batch_order_hash[:12]}",
              flush=True)

    return model, history, epoch_records, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["specialist"], required=True)
    parser.add_argument("--load-ckpt", required=True,
                         help="warm-start checkpoint；最终 B 榜配方传 ep149 base")
    parser.add_argument("--firstpass-cache", required=True,
                         help="ep149 对训练 datalist 生成的 merged cache pred/ 目录")
    parser.add_argument("--clean-dir", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--train-datalist", required=True)
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patches-per-sample", type=int, default=N_PATCHES_PER_SAMPLE)
    parser.add_argument("--epochs", type=int, default=0, help="0=用 stage 默认值")
    parser.add_argument("--lr", type=float, default=0.0, help="0=用 stage 默认值")
    parser.add_argument("--batch-size", type=int, default=0, help="0=用 stage 默认值")
    parser.add_argument(
        "--mlgc-hidden",
        type=int,
        default=64,
        help="必须与 warm-start checkpoint 的 MLGC hidden 宽度一致；最终方案默认 64",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--run-id", default="",
                         help="输出 run 目录名；留空使用时间戳")
    parser.add_argument("--smoke", action="store_true",
                         help="冒烟模式：--epochs 1 --patches-per-sample 1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jt.set_global_seed(args.seed)

    defaults = STAGE_DEFAULTS[args.stage]
    epochs = args.epochs or defaults["epochs"]
    lr = args.lr or defaults["lr"]
    batch_size = args.batch_size or defaults["batch_size"]

    train_datalist = rel_path(REPO_ROOT, args.train_datalist)
    clean_root = rel_path(REPO_ROOT, args.clean_dir)
    load_ckpt = rel_path(REPO_ROOT, args.load_ckpt)
    firstpass_root = rel_path(REPO_ROOT, args.firstpass_cache)
    rels = read_datalist(train_datalist)

    patch_seed = args.seed
    shuffle_seed = args.seed + 10_000_000

    print(f"[B-FINAL-{args.stage.upper()}] specialist 训练（最终 patch 契约）")
    print(f"  script_version: {SCRIPT_VERSION}")
    print(f"  stage: {args.stage}")
    print(f"  load_ckpt (warm-start): {load_ckpt}")
    print(f"  firstpass_cache: {firstpass_root}")
    print(f"  clean_dir: {clean_root}")
    print(f"  train_datalist: {train_datalist} (n={len(rels)})")
    print(f"  epochs={epochs}, lr={lr}, batch={batch_size}, "
          f"patches_per_sample={args.patches_per_sample}")
    print(f"  mlgc_hidden={args.mlgc_hidden}")
    print(f"  patch_seed={patch_seed} shuffle_seed={shuffle_seed}")
    if args.smoke:
        epochs = 1
        args.patches_per_sample = 1
        print("  [SMOKE MODE] epochs=1 patches_per_sample=1")

    missing = []
    for rel in rels:
        if not (firstpass_root / rel / "denoised.npy").exists():
            missing.append(f"firstpass:{rel}")
        if not (clean_root / rel / "clean.npy").exists():
            missing.append(f"clean:{rel}")
    if missing:
        raise SystemExit(f"[FAIL] 缺少必要文件: {missing[:10]} (共 {len(missing)} 个)")

    print(f"\n[PHASE 1] 加载 warm-start ckpt...")
    model = load_model(load_ckpt, mlgc_hidden=args.mlgc_hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    print(f"\n[PHASE 2] 训练 ({epochs} ep, lr={lr}, batch={batch_size})...")
    t0 = time.time()
    model, history, epoch_records, total_steps = train_c_contract(
        model, rels, firstpass_root, clean_root, epochs,
        args.patches_per_sample, args.patch_size, lr, batch_size,
        patch_seed, shuffle_seed,
    )
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s ({train_time / 60:.1f}min), total_steps={total_steps}")

    run_id = args.run_id or (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_b_final_{args.stage}_specialist_train"
    )
    out_dir = REPO_ROOT / "outputs" / "runs" / VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_out = out_dir / "b_specialist_final.pkl"
    state_online = {"network." + k: v for k, v in model.state_dict().items()}
    jt.save(state_online, str(ckpt_out))

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/experiments/b_final/train_deep_specialist.py",
        "script_version": SCRIPT_VERSION,
        "stage": args.stage,
        "contract": "final_patch_contract (每 epoch 重采 + FPS seed 点中心化)",
        "load_ckpt": str(load_ckpt),
        "firstpass_cache": str(firstpass_root),
        "clean_dir": str(clean_root),
        "train_datalist": str(train_datalist),
        "ckpt_raw": str(ckpt_out),
        "n_train_samples": len(rels),
        "patches_per_sample": args.patches_per_sample,
        "patch_size": args.patch_size,
        "n_epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "mlgc_hidden": args.mlgc_hidden,
        "seed": args.seed,
        "patch_seed": patch_seed,
        "shuffle_seed": shuffle_seed,
        "ema_enabled": False,
        "total_optimizer_steps": total_steps,
        "train_time_sec": round(train_time, 1),
        "loss_history": history,
        "epoch_records": epoch_records,
        "note": "warm-start from B ep149; final evaluation uses the public B final cascade task.",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[PHASE 3] ckpt saved: {ckpt_out}")
    print(f"[DONE] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
