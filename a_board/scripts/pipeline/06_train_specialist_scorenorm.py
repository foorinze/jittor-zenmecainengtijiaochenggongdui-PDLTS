#!/usr/bin/env python3
"""第二阶段第三步：评分归一化损失微调。

动机来自官方评分公式本身。官方对每个样本算的是相对改善比例：

    cd_score_i = 100 * (1 - CD_pred_i / CD_noisy_i)

分母是该样本 noisy 输入自身的误差。也就是说，同样大小的绝对误差下降，
在低噪声样本上换来的分数比高噪声样本更多。而绝对 Chamfer 损失对所有样本
一视同仁，梯度天然被高噪声样本主导——低噪声样本的每单位改善更值钱，却
拿到更少的梯度权重。

修正做法：给每个样本的损失乘一个与其噪声水平成反比的权重

    w_raw_i  = clip(median(CD_noisy) / CD_noisy_i, 0.5, 3.0)
    w_norm_i = w_raw_i / E[w_raw]

权重在 2000 个训练样本的总体上一次性预计算（由
04_precompute_score_weights.py 产出），训练时只查表，不做 batch 内动态
归一化——batch 内归一化会让同一样本的权重随同批样本变化，破坏可复现性。
除以 E[w_raw] 是为了让加权后损失的整体尺度与加权前一致，避免顺带改变有效
学习率。

clip 上下界 0.5/3.0 限制单样本权重的极端值，防止个别极低噪声样本主导梯度。

本步唯一变量就是这个损失权重：热启动上一步产物，数据、patch 采样、epoch 数、
学习率、batch、seed 全部与上一步逐项一致。

用法：
    python scripts/pipeline/06_train_specialist_scorenorm.py \\
        --train-datalist starter_code/datalist/specialist_train2000.txt \\
        --firstpass-cache <第一阶段对 2000 样本的输出 merged pred/ 目录> \\
        --load-ckpt outputs/runs/a_final/repro_specialist_step2_finetune/specialist_step2_finetune.pkl \\
        --cd-noisy-sidecar outputs/diagnostics/a_final/repro_score_weights/cd_noisy_sidecar.json \\
        --run-id repro_specialist_step3_scorenorm
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

EPOCHS = 12
LR = 5e-5
BATCH_SIZE = 16
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
    from scipy.spatial import cKDTree

    seeds = fps_indices(input_pts, n_patches, patch_rng)
    tree = cKDTree(input_pts)
    patches_in, patches_cl = [], []
    for seed_idx in seeds:
        _, idx = tree.query(input_pts[seed_idx], k=patch_size)
        patch_in = input_pts[idx]
        patch_cl = clean_pts[idx]
        center = input_pts[seed_idx]
        patches_in.append((patch_in - center).astype(np.float32))
        patches_cl.append((patch_cl - center).astype(np.float32))
    return np.array(patches_in), np.array(patches_cl), seeds


def extract_epoch_patches_weighted(
    rels: list[str],
    firstpass_root: Path,
    clean_root: Path,
    n_patches_per_sample: int,
    patch_size: int,
    patch_rng: np.random.RandomState,
    w_norm_map: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """与 05_train_specialist.py 的 patch 抽取一致，额外返回逐 patch 的样本级权重。"""
    all_in, all_cl, all_seeds, all_w = [], [], [], []
    for rel in rels:
        fp = np.load(firstpass_root / rel / "denoised.npy").astype(np.float32)
        cl = np.load(clean_root / rel / "clean.npy").astype(np.float32)
        pi, pc, seeds = extract_patches_one_shape(
            fp, cl, n_patches_per_sample, patch_size, patch_rng)
        all_in.append(pi)
        all_cl.append(pc)
        all_seeds.append(seeds)
        w = w_norm_map[rel]
        all_w.append(np.full(len(pi), w, dtype=np.float32))
    all_in = np.concatenate(all_in, axis=0)
    all_cl = np.concatenate(all_cl, axis=0)
    all_w = np.concatenate(all_w, axis=0)
    concat_seeds = np.concatenate(all_seeds).astype(np.int64)
    patch_index_hash = hashlib.sha256(concat_seeds.tobytes()).hexdigest()
    return all_in, all_cl, all_w, patch_index_hash


def load_model(ckpt_path: Path):
    from src.model.pdlts_light.model import PDLTSLightNetwork

    model = PDLTSLightNetwork()
    state = jt.load(str(ckpt_path))
    if any(k.startswith("network.") for k in state):
        state = {k.replace("network.", ""): v for k, v in state.items()
                 if k.startswith("network.")}
    model.load_state_dict(state)
    return model


def chamfer_l2_per_sample(pred: jt.Var, target: jt.Var) -> jt.Var:
    """返回逐 sample（不 reduce 到标量）的对称 Chamfer L2，形状 (batch,)。

    与 05_train_specialist.py 里 chamfer_l2 的唯一区别：那个函数对 batch 做 .mean()
    直接归约到标量；这里保留 batch 维，让调用方先乘 shape 级权重再归约，
    确保权重作用在"每个 patch 的 loss"上而不是作用在已经被均值坍缩过的
    batch 标量上（后者会让权重退化成对整个 batch 的一个缩放，语义不对）。
    """
    batch, n_pred, _ = pred.shape
    _, n_target, _ = target.shape
    _, idx_p2t = jt.misc.knn(pred, target, 1)
    bi = jt.arange(batch).view(-1, 1)
    nn_t = target[bi, idx_p2t.reshape(batch, n_pred)]
    d_p2t = ((pred - nn_t) ** 2).sum(dim=-1).mean(dim=-1)  # (batch,)

    _, idx_t2p = jt.misc.knn(target, pred, 1)
    nn_p = pred[bi, idx_t2p.reshape(batch, n_target)]
    d_t2p = ((target - nn_p) ** 2).sum(dim=-1).mean(dim=-1)  # (batch,)
    return d_p2t + d_t2p


def train_scorenorm(
    model,
    rels: list[str],
    firstpass_root: Path,
    clean_root: Path,
    w_norm_map: dict[str, float],
    n_epochs: int,
    n_patches_per_sample: int,
    patch_size: int,
    lr: float,
    batch_size: int,
    patch_seed: int,
    shuffle_seed: int,
):
    optimizer = jt.optim.Adam(model.parameters(), lr=lr)
    patch_rng = np.random.RandomState(patch_seed)
    shuffle_rng = np.random.RandomState(shuffle_seed)

    epoch_records = []
    history = []
    global_step = 0

    for ep in range(n_epochs):
        model.train()
        all_in, all_cl, all_w, patch_index_hash = extract_epoch_patches_weighted(
            rels, firstpass_root, clean_root, n_patches_per_sample, patch_size,
            patch_rng, w_norm_map)

        n_total = len(all_in)
        perm = shuffle_rng.permutation(n_total)
        batch_order_hash = hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_total, batch_size):
            idx = perm[start:start + batch_size]
            inp = jt.array(all_in[idx])
            tgt = jt.array(all_cl[idx])
            w = jt.array(all_w[idx])
            per_sample_loss = chamfer_l2_per_sample(model(inp)[0], tgt)
            loss = (per_sample_loss * w).mean()
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


def run_smoke(model, rels, firstpass_root, clean_root, w_norm_map, patch_size, patch_seed):
    """1-batch smoke：加权 loss finite、梯度非零、w_norm 分布打印。"""
    patch_rng = np.random.RandomState(patch_seed)
    all_in, all_cl, all_w, _ = extract_epoch_patches_weighted(
        rels[:8], firstpass_root, clean_root, 1, patch_size, patch_rng, w_norm_map)

    optimizer = jt.optim.Adam(model.parameters(), lr=LR)
    inp = jt.array(all_in[:8])
    tgt = jt.array(all_cl[:8])
    w = jt.array(all_w[:8])
    per_sample_loss = chamfer_l2_per_sample(model(inp)[0], tgt)
    loss = (per_sample_loss * w).mean()
    loss_val = float(loss.item())
    finite = np.isfinite(loss_val)
    optimizer.step(loss)

    n_nonzero_grad = 0
    n_total_grad = 0
    for p in model.parameters():
        n_total_grad += 1
        try:
            g = p.opt_grad(optimizer)
        except RuntimeError:
            continue
        if g is None:
            continue
        if float(jt.abs(g).sum().item()) > 0:
            n_nonzero_grad += 1

    print(f"  [SMOKE] loss={loss_val:.8f} finite={finite}")
    print(f"  [SMOKE] grad coverage: {n_nonzero_grad}/{n_total_grad}")
    print(f"  [SMOKE] w batch: {all_w[:8]}")
    return finite, n_nonzero_grad, n_total_grad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-ckpt", required=True,
                         help="热启动权重：上一步（finetune）的产物")
    parser.add_argument("--firstpass-cache", required=True)
    parser.add_argument("--clean-dir", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--train-datalist", required=True)
    parser.add_argument("--cd-noisy-sidecar", required=True,
                         help="04_precompute_score_weights.py 产出的权重 json")
    parser.add_argument("--run-id", default="",
                         help="输出 run 目录名。留空则用时间戳")
    parser.add_argument("--patch-size", type=int, default=PATCH_SIZE)
    parser.add_argument("--patches-per-sample", type=int, default=N_PATCHES_PER_SAMPLE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--smoke", action="store_true",
                         help="1-batch smoke：验证 loss finite + 梯度非零 + w_norm 分布，不训练")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jt.set_global_seed(args.seed)

    train_datalist = rel_path(REPO_ROOT, args.train_datalist)
    clean_root = rel_path(REPO_ROOT, args.clean_dir)
    load_ckpt = rel_path(REPO_ROOT, args.load_ckpt)
    firstpass_root = rel_path(REPO_ROOT, args.firstpass_cache)
    sidecar_path = rel_path(REPO_ROOT, args.cd_noisy_sidecar)
    rels = read_datalist(train_datalist)

    with sidecar_path.open("r", encoding="utf-8") as f:
        sidecar = json.load(f)
    w_norm_map = sidecar["w_norm"]

    missing_w = [rel for rel in rels if rel not in w_norm_map]
    if missing_w:
        raise SystemExit(f"[FAIL] {len(missing_w)} 个训练样本在 CD_noisy sidecar 里缺权重: "
                          f"{missing_w[:5]}")

    patch_seed = args.seed
    shuffle_seed = args.seed + 10_000_000

    print("[精修网络-scorenorm] 评分归一化损失微调")
    print(f"  script_version: {SCRIPT_VERSION}")
    print(f"  load_ckpt (热启动): {load_ckpt}")
    print(f"  firstpass_cache: {firstpass_root}")
    print(f"  clean_dir: {clean_root}")
    print(f"  train_datalist: {train_datalist} (n={len(rels)})")
    print(f"  cd_noisy_sidecar: {sidecar_path}")
    print(f"  E[w_raw]={sidecar['e_w_raw']:.6f} median_cd_noisy={sidecar['median_cd_noisy']:.8f}")
    print(f"  epochs={args.epochs}, lr={args.lr}, batch={args.batch_size}, "
          f"patches_per_sample={args.patches_per_sample}")
    print(f"  patch_seed={patch_seed} shuffle_seed={shuffle_seed}")

    missing = []
    for rel in rels:
        if not (firstpass_root / rel / "denoised.npy").exists():
            missing.append(f"firstpass:{rel}")
        if not (clean_root / rel / "clean.npy").exists():
            missing.append(f"clean:{rel}")
    if missing:
        raise SystemExit(f"[FAIL] 缺少必要文件: {missing[:10]} (共 {len(missing)} 个)")

    print("\n[步骤 1] 加载热启动权重...")
    model = load_model(load_ckpt)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    if args.smoke:
        print("\n[步骤 2] 单 batch 冒烟检查...")
        finite, n_nonzero, n_total = run_smoke(
            model, rels, firstpass_root, clean_root, w_norm_map, args.patch_size, patch_seed)
        if not finite:
            raise SystemExit("[FAIL] smoke loss 非 finite")
        if n_nonzero == 0:
            raise SystemExit("[FAIL] smoke 梯度全零")
        print(f"[SMOKE_PASS] loss finite={finite}, grad_nonzero={n_nonzero}/{n_total}")
        return

    print(f"\n[步骤 2] 训练 ({args.epochs} ep, lr={args.lr}, batch={args.batch_size})...")
    t0 = time.time()
    model, history, epoch_records, total_steps = train_scorenorm(
        model, rels, firstpass_root, clean_root, w_norm_map, args.epochs,
        args.patches_per_sample, args.patch_size, args.lr, args.batch_size,
        patch_seed, shuffle_seed,
    )
    train_time = time.time() - t0
    print(f"  done in {train_time:.1f}s ({train_time / 60:.1f}min), total_steps={total_steps}")

    run_id = args.run_id or (
        time.strftime("%Y%m%d_%H%M%S") + "_a_final_specialist_scorenorm"
    )
    out_dir = REPO_ROOT / "outputs" / "runs" / VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_out = out_dir / "specialist_step3_scorenorm.pkl"
    state_online = {"network." + k: v for k, v in model.state_dict().items()}
    jt.save(state_online, str(ckpt_out))

    summary = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/pipeline/06_train_specialist_scorenorm.py",
        "script_version": SCRIPT_VERSION,
        "sampling": "每 epoch 重采 patch + 种子点中心化 + 评分归一化损失",
        "load_ckpt": str(load_ckpt),
        "firstpass_cache": str(firstpass_root),
        "clean_dir": str(clean_root),
        "train_datalist": str(train_datalist),
        "cd_noisy_sidecar": str(sidecar_path),
        "e_w_raw": sidecar["e_w_raw"],
        "median_cd_noisy": sidecar["median_cd_noisy"],
        "ckpt_raw": str(ckpt_out),
        "n_train_samples": len(rels),
        "patches_per_sample": args.patches_per_sample,
        "patch_size": args.patch_size,
        "n_epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "patch_seed": patch_seed,
        "shuffle_seed": shuffle_seed,
        "ema_enabled": False,
        "total_optimizer_steps": total_steps,
        "train_time_sec": round(train_time, 1),
        "loss_history": history,
        "epoch_records": epoch_records,
        "note": "热启动上一步产物。评测一律走 starter_code/run.py 的 predict 管线 + 官方 evaluate 脚本。",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[步骤 3] 权重已保存: {ckpt_out}")
    print(f"[DONE] summary -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
