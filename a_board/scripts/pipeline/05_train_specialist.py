#!/usr/bin/env python3
"""第二阶段（精修网络）训练：主训练 + 全量微调两步。

精修网络与第一阶段同架构，但训练目标不是「噪声 -> 真值」，而是
「第一阶段输出 -> 真值」的残余映射，权重从第一阶段热启动。

两步调度：

    --stage main      热启动第一阶段权重，在 1600 样本子集上训 50 epoch，
                      lr 1e-4 / batch 16。
    --stage finetune   热启动 main 产物，在全量 2000 样本上训 12 epoch，
                      lr 5e-5 / batch 16。

两步都采用训练-推理一致性采样：
    1. 每个 epoch 重新抽 patch（不是全程复用第一个 epoch 的固定 patch 集），
       让有效数据覆盖随 epoch 增长；
    2. patch 中心用最远点采样得到的种子点本身，而不是 patch 内点的均值。
       推理时的 patch 中心就是种子点，用均值会让训练与推理的输入分布错配。

随机性全部由 seed 决定，patch 抽样与 batch 打乱用两个独立的随机数发生器
（patch_seed = seed，shuffle_seed = seed + 10000000），避免一方的调用次数
变化影响另一方的序列。每个 epoch 落盘 patch 索引与 batch 顺序的 sha256，
便于核对两次运行是否逐位一致。

用法：
    python scripts/pipeline/05_train_specialist.py --stage main \\
        --train-datalist starter_code/datalist/specialist_train1600.txt \\
        --firstpass-cache <第一阶段对 1600 样本的输出 pred/ 目录> \\
        --load-ckpt outputs/runs/a_final/repro_base_train/checkpoints/pdlts_light_99.pkl \\
        --run-id repro_specialist_step1_main

    python scripts/pipeline/05_train_specialist.py --stage finetune \\
        --train-datalist starter_code/datalist/specialist_train2000.txt \\
        --firstpass-cache <第一阶段对 2000 样本的输出 merged pred/ 目录> \\
        --load-ckpt outputs/runs/a_final/repro_specialist_step1_main/specialist_step1_main.pkl \\
        --run-id repro_specialist_step2_finetune
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
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "starter_code"))

import jittor as jt  # noqa: E402

jt.flags.use_cuda = 1

VERSION = "a_final"
SCRIPT_VERSION = "public"

DEFAULT_TRAIN_ROOT = "dataset/a_final_train_full15k"
DEFAULT_BASE_CKPT = (
    "outputs/runs/a_final/repro_base_train/checkpoints/pdlts_light_99.pkl"
)

STAGE_DEFAULTS = {
    "main": {"epochs": 50, "lr": 1e-4, "batch_size": 16},
    "finetune": {"epochs": 12, "lr": 5e-5, "batch_size": 16},
}
STAGE_CKPT_NAME = {
    "main": "specialist_step1_main",
    "finetune": "specialist_step2_finetune",
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
    """最远点采样：返回 n_seeds 个尽量分散的点索引，作为 patch 中心。"""
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
    """切 patch。中心取最远点采样得到的种子点本身，与推理口径一致。"""
    from scipy.spatial import cKDTree

    seeds = fps_indices(input_pts, n_patches, patch_rng)
    tree = cKDTree(input_pts)
    patches_in, patches_cl = [], []
    for seed_idx in seeds:
        _, idx = tree.query(input_pts[seed_idx], k=patch_size)
        patch_in = input_pts[idx]
        patch_cl = clean_pts[idx]
        center = input_pts[seed_idx]  # 中心 = 种子点本身，不是 patch 均值
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


def load_model(ckpt_path: Path):
    from src.model.pdlts_light.model import PDLTSLightNetwork

    model = PDLTSLightNetwork()
    state = jt.load(str(ckpt_path))
    if any(k.startswith("network.") for k in state):
        state = {k.replace("network.", ""): v for k, v in state.items()
                 if k.startswith("network.")}
    model.load_state_dict(state)
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


def train_specialist(
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
    """训练循环：每 epoch 重采 patch + 种子点中心化。"""
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
    parser.add_argument("--stage", choices=["main", "finetune"], required=True)
    parser.add_argument("--load-ckpt", default="",
                         help="热启动权重。main 默认用第一阶段 ep99；"
                              "finetune 必须显式传 main 步的产物")
    parser.add_argument("--firstpass-cache", required=True,
                         help="第一阶段对本步 datalist 跑出的输出 pred/ 目录"
                              "（由 03_generate_firstpass_cache.py 生成并合并）")
    parser.add_argument("--clean-dir", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--train-datalist", required=True)
    parser.add_argument("--run-id", default="",
                         help="输出 run 目录名。留空则用时间戳 + stage 名")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patches-per-sample", type=int, default=N_PATCHES_PER_SAMPLE)
    parser.add_argument("--epochs", type=int, default=0, help="0=用该步默认值")
    parser.add_argument("--lr", type=float, default=0.0, help="0=用该步默认值")
    parser.add_argument("--batch-size", type=int, default=0, help="0=用该步默认值")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true",
                         help="冒烟模式：只跑 1 epoch、每样本 1 个 patch，验证链路通畅")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jt.set_global_seed(args.seed)

    defaults = STAGE_DEFAULTS[args.stage]
    epochs = args.epochs or defaults["epochs"]
    lr = args.lr or defaults["lr"]
    batch_size = args.batch_size or defaults["batch_size"]

    load_ckpt_str = args.load_ckpt
    if not load_ckpt_str:
        if args.stage == "main":
            load_ckpt_str = DEFAULT_BASE_CKPT
        else:
            raise SystemExit(
                "[FAIL] --stage finetune 必须显式传 --load-ckpt（main 步的产物）"
            )

    train_datalist = rel_path(REPO_ROOT, args.train_datalist)
    clean_root = rel_path(REPO_ROOT, args.clean_dir)
    load_ckpt = rel_path(REPO_ROOT, load_ckpt_str)
    firstpass_root = rel_path(REPO_ROOT, args.firstpass_cache)
    rels = read_datalist(train_datalist)

    patch_seed = args.seed
    shuffle_seed = args.seed + 10_000_000

    print(f"[精修网络-{args.stage}] 训练开始")
    print(f"  script_version: {SCRIPT_VERSION}")
    print(f"  stage: {args.stage}")
    print(f"  load_ckpt (热启动): {load_ckpt}")
    print(f"  firstpass_cache: {firstpass_root}")
    print(f"  clean_dir: {clean_root}")
    print(f"  train_datalist: {train_datalist} (n={len(rels)})")
    print(f"  epochs={epochs}, lr={lr}, batch={batch_size}, "
          f"patches_per_sample={args.patches_per_sample}")
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

    print(f"\n[步骤 1] 加载热启动权重...")
    model = load_model(load_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    print(f"\n[步骤 2] 训练 ({epochs} ep, lr={lr}, batch={batch_size})...")
    t0 = time.time()
    model, history, epoch_records, total_steps = train_specialist(
        model, rels, firstpass_root, clean_root, epochs,
        args.patches_per_sample, args.patch_size, lr, batch_size,
        patch_seed, shuffle_seed,
    )
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s ({train_time / 60:.1f}min), total_steps={total_steps}")

    run_id = args.run_id or (
        time.strftime("%Y%m%d_%H%M%S") + f"_a_final_specialist_{args.stage}"
    )
    out_dir = REPO_ROOT / "outputs" / "runs" / VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_out = out_dir / f"{STAGE_CKPT_NAME[args.stage]}.pkl"
    state_online = {"network." + k: v for k, v in model.state_dict().items()}
    jt.save(state_online, str(ckpt_out))

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/pipeline/05_train_specialist.py",
        "script_version": SCRIPT_VERSION,
        "stage": args.stage,
        "sampling": "每 epoch 重采 patch + 种子点中心化",
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
        "seed": args.seed,
        "patch_seed": patch_seed,
        "shuffle_seed": shuffle_seed,
        "ema_enabled": False,
        "total_optimizer_steps": total_steps,
        "train_time_sec": round(train_time, 1),
        "loss_history": history,
        "epoch_records": epoch_records,
        "note": "热启动训练。评测一律走 starter_code/run.py 的 predict 管线 + "
                "官方 evaluate 脚本，不在本脚本内自评。",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[步骤 3] 权重已保存: {ckpt_out}")
    print(f"[DONE] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
