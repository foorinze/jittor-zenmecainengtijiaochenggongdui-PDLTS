#!/usr/bin/env python3
"""A/B 榜 FPS+KNN 覆盖率扫描。

对 dataset/mock_test 下的样本扫 seed_k ∈ {5, 8, 10, 12, 16} 的 FPS+KNN 覆盖率,
**不 forward network**, 纯几何覆盖度量.

设计约束:
  1. 覆盖算法必须和 ``patch_denoise`` 的覆盖判定完全一致:
       K = ceil(seed_k * N / effective_patch_size)
       effective_patch_size = min(patch_size, N)
       FPS + jt.misc.knn 取邻居, is_covered = any patch 里出现过 p
  2. CSV schema 至少: rel_path, cls_id, object_id, seed_k, patch_size,
     N, K, effective_patch_size, n_covered, n_missing, missing_ratio,
     coverage_ratio, fps_sec, knn_sec, total_sec, status, error
  3. workers = 1 串行；>1 直接 fail
  4. resume 去重键 = (rel_path, seed_k, patch_size) 三元组
  5. manifest.json 记录 jittor_version, use_cuda, git_commit, script_version,
     seed_k_list, patch_size, datalist, mock_dir, created_at, aggregate_summary
  6. 归一化走 dataset/mock_test 每个样本的 norm.json。注意: coverage 量 (FPS+KNN) 对平移和统一缩放不敏感, 因此 norm.json 和推理时 bbox 自归一化对 coverage 度量结果等价。但措辞上写"对 coverage 等价"而非"完全同推理归一化"。
     没有 norm.json 的样本 status='skipped_no_norm'

使用 (从项目根或 starter_code/ 都行):
  python scripts/shared/coverage_sweep.py \
      --stage b_final                         \
      --datalist starter_code/datalist/mock.txt  \
      --mock-dir dataset/mock_test               \
      --seed-k-list 5,8,10,12,16                 \
      --eval-id sweep_subset_20                  \
      [--patch-size 1024]                         \
      [--limit N]                                 \
      [--resume]                                  \
      [--workers 1]
"""

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

def _find_project_root() -> Path:
    """从脚本位置向上寻找公开代码根目录。"""
    for parent in Path(__file__).resolve().parents:
        if (parent / "starter_code").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("无法定位包含 starter_code/ 和 scripts/ 的代码根目录")


_PROJECT_ROOT = _find_project_root()
for _path in (_PROJECT_ROOT / "scripts" / "shared", _PROJECT_ROOT / "scripts", _PROJECT_ROOT / "starter_code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np


SCRIPT_VERSION = "public"


def _resolve_path(p: str, project_root: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(project_root, p))


def _git_commit() -> dict:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        if r.returncode != 0:
            return {"status": "not-a-repo"}
        commit = r.stdout.strip()
        branch = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip() or "UNKNOWN"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip()
        return {
            "status": "ok",
            "commit": commit,
            "branch": branch,
            "dirty": "yes" if dirty else "no",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ---------------------------------------------------------------------------
# 覆盖率核心算法：必须和 patch_denoise 对齐。
# ---------------------------------------------------------------------------
def compute_coverage(pcl_noisy_np: np.ndarray, patch_size: int, seed_k: int):
    """计算 FPS+KNN 覆盖率 (不 forward network).

    严格对齐 starter_code/src/model/pdlts_light/denoise.py:patch_denoise 的覆盖判定:
        K = max(1, ceil(seed_k * N / effective_patch_size))
        effective_patch_size = min(patch_size, N)
        FPS 采 K 个 seed, 对每个 seed 用 KNN 取 effective_patch_size 个邻居,
        被至少一个 patch 覆盖的点记作 covered.

    Args:
        pcl_noisy_np: (N, 3) float32, **已归一化**点云 (和 patch_denoise 调用时一致)
        patch_size: 每 patch 点数
        seed_k: 平均覆盖次数

    Returns:
        dict, 含:
            N, K, effective_patch_size
            n_covered, n_missing, coverage_ratio, missing_ratio
            fps_sec, knn_sec, total_sec
    """
    # 延迟 import: 允许脚本在没有 jittor 的环境下做 --help / resume 检查.
    import jittor as jt

    assert pcl_noisy_np.ndim == 2 and pcl_noisy_np.shape[-1] == 3, \
        f"expected (N, 3), got {pcl_noisy_np.shape}"
    N = pcl_noisy_np.shape[0]
    effective_patch_size = min(int(patch_size), int(N))
    K = max(1, math.ceil(seed_k * N / effective_patch_size))

    t0 = time.time()
    pcl_noisy = jt.array(pcl_noisy_np.astype(np.float32))
    pcl_noisy_b = pcl_noisy.unsqueeze(0)

    # --- FPS ---
    t_fps_start = time.time()
    from src.model.vm import farthest_point_sampling
    seed_pnts, _ = farthest_point_sampling(pcl_noisy_b, K)
    jt.sync_all()  # 等 Jittor 实际算完, 别让 lazy eval 把时间算到下一步
    fps_sec = time.time() - t_fps_start

    # --- KNN ---
    t_knn_start = time.time()
    _, point_idxs = jt.misc.knn(seed_pnts, pcl_noisy_b, effective_patch_size)
    point_idxs_np = point_idxs[0].numpy().astype(np.int64)  # (K, effective_patch_size)
    knn_sec = time.time() - t_knn_start

    # --- 覆盖判定 (严格同 patch_denoise: 出现在任何 patch 的 idx 里即覆盖) ---
    covered = np.zeros(N, dtype=bool)
    for k in range(K):
        covered[point_idxs_np[k]] = True
    n_covered = int(covered.sum())
    n_missing = int(N - n_covered)
    total_sec = time.time() - t0

    return {
        "N": N,
        "K": K,
        "effective_patch_size": effective_patch_size,
        "n_covered": n_covered,
        "n_missing": n_missing,
        "coverage_ratio": float(n_covered) / float(N),
        "missing_ratio": float(n_missing) / float(N),
        "fps_sec": round(fps_sec, 4),
        "knn_sec": round(knn_sec, 4),
        "total_sec": round(total_sec, 4),
    }


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
CSV_COLUMNS = [
    "rel_path", "cls_id", "object_id",
    "seed_k", "patch_size",
    "N", "K", "effective_patch_size",
    "n_covered", "n_missing", "missing_ratio", "coverage_ratio",
    "fps_sec", "knn_sec", "total_sec",
    "status", "error",
]


def _load_datalist(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def _load_noisy_and_norm(mock_dir: str, rel_path: str) -> Optional[Tuple[np.ndarray, Optional[dict]]]:
    """加载 mock_test 下 rel_path 的 noisy + norm.json. 返回 (noisy_np, norm) 或 None 表缺失."""
    sample_dir = os.path.join(mock_dir, rel_path)
    noisy_path = os.path.join(sample_dir, "noisy.npy")
    norm_path = os.path.join(sample_dir, "norm.json")
    if not os.path.exists(noisy_path):
        return None, None, "missing_noisy"
    try:
        noisy = np.load(noisy_path).astype(np.float32)
    except Exception as e:
        return None, None, f"load_noisy_err:{e}"
    norm = None
    if os.path.exists(norm_path):
        try:
            with open(norm_path, "r", encoding="utf-8") as f:
                norm = json.load(f)
        except Exception:
            norm = None
    return noisy, norm, ""


def _normalize_with_norm(pc: np.ndarray, norm: dict) -> np.ndarray:
    center = np.asarray(norm["center"], dtype=np.float32)
    scale = float(norm["scale"])
    return ((pc - center) / max(scale, 1e-12)).astype(np.float32)


def _read_existing_keys(csv_path: str) -> set:
    """读已有 CSV, 返回已完成的 (rel_path, seed_k, patch_size) 集合 (status in {ok, skipped_no_norm})."""
    keys = set()
    if not os.path.exists(csv_path):
        return keys
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            status = row.get("status", "")
            if status in ("ok", "skipped_no_norm"):
                try:
                    keys.add((row["rel_path"], int(row["seed_k"]), int(row["patch_size"])))
                except Exception:
                    pass
    return keys


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datalist", required=True,
                   help="datalist.txt 每行 'shapenet/<cls>/<id>' (相对 mock_dir)")
    p.add_argument("--mock-dir", default="dataset/mock_test",
                   help="mock_test 根目录 (相对项目根或绝对)")
    p.add_argument("--seed-k-list", default="5,8,10,12,16",
                   help="seed_k 档位, 逗号分隔")
    p.add_argument("--patch-size", type=int, default=1024,
                   help="patch_size, 默认 Light 规格")
    p.add_argument("--stage", required=True, choices=("a_final", "b_final"),
                   help="公开阶段标签；决定产物写入的阶段目录")
    p.add_argument("--eval-id", default=None,
                   help="eval run id；默认时间戳加阶段和 coverage_sweep")
    p.add_argument("--eval-root", default="outputs/evals",
                   help="eval 根目录 (相对项目根或绝对)")
    p.add_argument("--limit", type=int, default=0,
                   help="只扫前 N 样本 (0=全部)")
    p.add_argument("--resume", action="store_true",
                   help="读已有 CSV, 跳过已完成的 (rel_path, seed_k, patch_size)")
    p.add_argument("--workers", type=int, default=1,
                   help="必须是 1；大于 1 直接失败")
    args = p.parse_args()

    if args.workers > 1:
        sys.exit(f"[FAIL] --workers > 1 is not supported (got {args.workers}); "
                 f"Jittor + CUDA + multiprocessing 容易踩坑, 先串行跑稳")

    root = _PROJECT_ROOT
    datalist_path = _resolve_path(args.datalist, root)
    mock_dir = _resolve_path(args.mock_dir, root)
    eval_root = Path(_resolve_path(args.eval_root, root))

    if not os.path.exists(datalist_path):
        sys.exit(f"[FAIL] datalist not found: {datalist_path}")
    if not os.path.isdir(mock_dir):
        sys.exit(f"[FAIL] mock_dir not a directory: {mock_dir}")

    samples = _load_datalist(datalist_path)
    if args.limit > 0:
        samples = samples[:args.limit]
    if not samples:
        sys.exit(f"[FAIL] datalist is empty")

    seed_k_list = [int(x) for x in args.seed_k_list.split(",") if x.strip()]
    if not seed_k_list:
        sys.exit("[FAIL] empty --seed-k-list")

    eval_id = args.eval_id or (
        time.strftime("%Y%m%d_%H%M%S") + f"_{args.stage}_coverage_sweep"
    )
    eval_dir = str(eval_root / args.stage / eval_id)
    os.makedirs(os.path.join(eval_dir, "logs"), exist_ok=True)

    # sys.path 准备 (允许 from src.model.vm import farthest_point_sampling)
    starter_code = os.path.join(root, "starter_code")
    if starter_code not in sys.path:
        sys.path.insert(0, starter_code)

    # Jittor 初始化
    import jittor as jt
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass

    # Resume
    csv_path = os.path.join(eval_dir, "coverage_sweep.csv")
    existing_keys = _read_existing_keys(csv_path) if args.resume else set()

    # manifest 记录运行环境和输入契约。
    manifest = {
        "script": "scripts/shared/coverage_sweep.py",
        "script_version": SCRIPT_VERSION,
        "stage": args.stage,
        "eval_id": eval_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmdline": sys.argv,
        "args": {
            "stage": args.stage,
            "datalist": datalist_path,
            "mock_dir": mock_dir,
            "seed_k_list": seed_k_list,
            "patch_size": args.patch_size,
            "limit": args.limit,
            "resume": args.resume,
            "workers": args.workers,
        },
        "env": {
            "jittor_version": jt.__version__,
            "use_cuda": int(jt.flags.use_cuda),
            "python": sys.version.splitlines()[0],
            "numpy": np.__version__,
        },
        "git_commit": _git_commit(),
        "samples_total": len(samples),
        "seed_k_total": len(seed_k_list),
        "combinations_total": len(samples) * len(seed_k_list),
    }
    with open(os.path.join(eval_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # 命令快照
    with open(os.path.join(eval_dir, "command.sh"), "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write(" ".join(repr(s) if " " in s else s for s in sys.argv) + "\n")

    # CSV: append 模式, 第一次写 header
    need_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    # 逐样本 × seed_k 写入 CSV，保证中途终止时已有可追溯记录。
    log_path = os.path.join(eval_dir, "logs", "sweep.log")
    with open(csv_path, "a", encoding="utf-8", newline="") as csvf, \
         open(log_path, "a", encoding="utf-8") as logf:
        writer = csv.DictWriter(csvf, fieldnames=CSV_COLUMNS)
        if need_header:
            writer.writeheader()
            csvf.flush()

        total_combos = len(samples) * len(seed_k_list)
        done = 0
        skipped_resume = 0
        errors = 0

        for rel_path in samples:
            parts = rel_path.split("/")
            if len(parts) < 3:
                # 非 shapenet/<cls>/<id> 格式, 记 error 并跳过
                cls_id, object_id = "", rel_path
            else:
                cls_id, object_id = parts[-2], parts[-1]

            # 加载 noisy + norm (一次, 所有 seed_k 共用)
            noisy, norm, err = _load_noisy_and_norm(mock_dir, rel_path)
            if noisy is None:
                for seed_k in seed_k_list:
                    key = (rel_path, seed_k, args.patch_size)
                    if key in existing_keys:
                        skipped_resume += 1
                        done += 1
                        continue
                    writer.writerow({
                        "rel_path": rel_path, "cls_id": cls_id, "object_id": object_id,
                        "seed_k": seed_k, "patch_size": args.patch_size,
                        "N": "", "K": "", "effective_patch_size": "",
                        "n_covered": "", "n_missing": "", "missing_ratio": "", "coverage_ratio": "",
                        "fps_sec": "", "knn_sec": "", "total_sec": "",
                        "status": "error_load", "error": err,
                    })
                    csvf.flush()
                    errors += 1
                    done += 1
                continue

            # norm.json 缺失 -> 记 skipped_no_norm (不用粗糙 bbox 归一化, 会污染决策)
            if norm is None:
                for seed_k in seed_k_list:
                    key = (rel_path, seed_k, args.patch_size)
                    if key in existing_keys:
                        skipped_resume += 1
                        done += 1
                        continue
                    writer.writerow({
                        "rel_path": rel_path, "cls_id": cls_id, "object_id": object_id,
                        "seed_k": seed_k, "patch_size": args.patch_size,
                        "N": int(noisy.shape[0]), "K": "", "effective_patch_size": "",
                        "n_covered": "", "n_missing": "", "missing_ratio": "", "coverage_ratio": "",
                        "fps_sec": "", "knn_sec": "", "total_sec": "",
                        "status": "skipped_no_norm", "error": "",
                    })
                    csvf.flush()
                    done += 1
                continue

            # 归一化 (走 norm.json; 对 coverage 等价, 不写"完全同推理归一化")
            normed = _normalize_with_norm(noisy, norm)

            for seed_k in seed_k_list:
                key = (rel_path, seed_k, args.patch_size)
                if key in existing_keys:
                    skipped_resume += 1
                    done += 1
                    continue

                try:
                    info = compute_coverage(normed, patch_size=args.patch_size, seed_k=seed_k)
                    writer.writerow({
                        "rel_path": rel_path, "cls_id": cls_id, "object_id": object_id,
                        "seed_k": seed_k, "patch_size": args.patch_size,
                        "N": info["N"], "K": info["K"],
                        "effective_patch_size": info["effective_patch_size"],
                        "n_covered": info["n_covered"], "n_missing": info["n_missing"],
                        "missing_ratio": info["missing_ratio"],
                        "coverage_ratio": info["coverage_ratio"],
                        "fps_sec": info["fps_sec"], "knn_sec": info["knn_sec"],
                        "total_sec": info["total_sec"],
                        "status": "ok", "error": "",
                    })
                    csvf.flush()
                except Exception as e:
                    writer.writerow({
                        "rel_path": rel_path, "cls_id": cls_id, "object_id": object_id,
                        "seed_k": seed_k, "patch_size": args.patch_size,
                        "N": int(noisy.shape[0]), "K": "", "effective_patch_size": "",
                        "n_covered": "", "n_missing": "", "missing_ratio": "", "coverage_ratio": "",
                        "fps_sec": "", "knn_sec": "", "total_sec": "",
                        "status": "error_compute", "error": str(e),
                    })
                    csvf.flush()
                    errors += 1

                done += 1
                if done % 10 == 0 or done == total_combos:
                    msg = (f"[{done}/{total_combos}] sample={rel_path} "
                           f"seed_k={seed_k} resume_skipped={skipped_resume} errors={errors}")
                    print(msg)
                    logf.write(msg + "\n")
                    logf.flush()

    # 聚合 summary 追写 manifest
    summary = _aggregate_summary(csv_path, seed_k_list, args.patch_size)
    with open(os.path.join(eval_dir, "manifest.json"), "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["aggregate_summary"] = summary
    with open(os.path.join(eval_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"[OK] coverage sweep done -> {eval_dir}")
    print(f"     combos={len(samples) * len(seed_k_list)}  "
          f"resume_skipped={skipped_resume}  errors={errors}")
    print(f"     summary:")
    for row in summary.get("per_seed_k", []):
        print(f"     seed_k={row['seed_k']:>3}  zero_miss={row['zero_miss_samples']}/"
              f"{row['ok_samples']}  max_mr={row['max_missing_ratio']:.2e}  "
              f"p95_mr={row['p95_missing_ratio']:.2e}  "
              f"avg_fps={row.get('avg_fps_sec', 0):.3f}s  "
              f"avg_knn={row.get('avg_knn_sec', 0):.3f}s  "
              f"avg_total={row['avg_total_sec']:.2f}s")
    print("=" * 60)


def _aggregate_summary(csv_path: str, seed_k_list: List[int], patch_size: int) -> dict:
    """按 seed_k 聚合 zero_miss / max_missing_ratio / p95 / avg_total_sec + status_counts."""
    per_seed_k = []
    if not os.path.exists(csv_path):
        return {"per_seed_k": [], "status_counts_total": {}}
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                if int(r["patch_size"]) != patch_size:
                    continue
                rows.append(r)
            except Exception:
                continue

    all_statuses = ("ok", "skipped_no_norm", "error_load", "error_compute")

    for sk in seed_k_list:
        rows_sk = [r for r in rows if int(r["seed_k"]) == sk]
        status_counts_sk = {s: 0 for s in all_statuses}
        for r in rows_sk:
            st = r.get("status", "")
            if st in status_counts_sk:
                status_counts_sk[st] += 1
            else:
                status_counts_sk.setdefault("other", 0)
                status_counts_sk["other"] += 1

        ok = [r for r in rows_sk if r["status"] == "ok"]
        if not ok:
            per_seed_k.append({
                "seed_k": sk, "ok_samples": 0, "zero_miss_samples": 0,
                "max_missing_ratio": 0.0, "p95_missing_ratio": 0.0,
                "mean_missing_ratio": 0.0,
                "avg_total_sec": 0.0, "avg_fps_sec": 0.0, "avg_knn_sec": 0.0,
                "status_counts": status_counts_sk,
            })
            continue
        mrs = np.array([float(r["missing_ratio"]) for r in ok])
        secs = np.array([float(r["total_sec"]) for r in ok])
        fps_arr = np.array([float(r["fps_sec"]) for r in ok])
        knn_arr = np.array([float(r["knn_sec"]) for r in ok])
        per_seed_k.append({
            "seed_k": sk,
            "ok_samples": len(ok),
            "zero_miss_samples": int((mrs == 0).sum()),
            "max_missing_ratio": float(mrs.max()),
            "p95_missing_ratio": float(np.percentile(mrs, 95)),
            "mean_missing_ratio": float(mrs.mean()),
            "avg_total_sec": float(secs.mean()),
            "avg_fps_sec": float(fps_arr.mean()),
            "avg_knn_sec": float(knn_arr.mean()),
            "status_counts": status_counts_sk,
        })

    # 总计 status_counts (跨所有 seed_k)
    status_counts_total = {s: 0 for s in all_statuses}
    for r in rows:
        st = r.get("status", "")
        if st in status_counts_total:
            status_counts_total[st] += 1
        else:
            status_counts_total.setdefault("other", 0)
            status_counts_total["other"] += 1

    return {
        "per_seed_k": per_seed_k,
        "status_counts_total": status_counts_total,
        "patch_size": patch_size,
    }


if __name__ == "__main__":
    main()
