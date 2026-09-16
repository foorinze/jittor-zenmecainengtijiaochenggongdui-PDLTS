#!/usr/bin/env python3
"""预计算评分归一化损失所需的逐样本权重。

第三步训练要给每个样本的损失乘一个与其噪声水平成反比的权重，权重依据是该
样本 noisy 输入自身的 Chamfer 误差 CD_noisy（也就是官方评分公式的分母）。

这些权重在 2000 个训练样本的总体上一次性算好写进 json，训练时只查表。
不在训练循环里现算，也不做 batch 内归一化：batch 内归一化会让同一个样本的
权重随同批样本变化，两次运行结果无法逐位复现。

口径一致性：CD_noisy 的定义必须与官方评分脚本完全一致，否则权重的物理含义
就不是"该样本每单位改善值多少分"。因此本脚本直接从 starter_code/evaluate_mock.py
import chamfer_distance 与 load_pointcloud，不另写一份 Chamfer 实现——
训练侧算权重和评测侧打分走的是同一份代码。

--smoke-check 会真实跑一次官方评测脚本的命令行，取它自己算出的 CD_noisy，
和本脚本直接调函数算出的值比对。验的不是 Chamfer 数学（同一个函数，不可能
不一致），而是加载方式是否一致：坐标系、点数、dtype 有没有处理偏差。

用法：
    python scripts/pipeline/04_precompute_score_weights.py \\
        --datalist starter_code/datalist/specialist_train2000.txt \\
        --train-root dataset/a_final_train_full15k \\
        --run-id repro_score_weights \\
        --smoke-check
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
STARTER_CODE = REPO_ROOT / "starter_code"
sys.path.insert(0, str(STARTER_CODE))

from evaluate_mock import chamfer_distance, load_pointcloud  # noqa: E402

VERSION = "a_final"
SCRIPT_VERSION = "public"

DEFAULT_TRAIN_ROOT = "dataset/a_final_train_full15k"
DEFAULT_MOCK_ROOT = "dataset/mock_test"
DEFAULT_MOCK_DATALIST = "starter_code/datalist/mock.txt"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_datalist(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8")
            if line.strip() and not line.startswith("#")]


def compute_cd_noisy_one(noisy_path: Path, clean_path: Path) -> float:
    """CD_noisy = chamfer_distance(noisy, clean, normalize=True)，与官方 evaluate_single 同一调用。"""
    pc_noisy = load_pointcloud(str(noisy_path))
    pc_clean = load_pointcloud(str(clean_path))
    return chamfer_distance(pc_noisy, pc_clean, normalize=True)


def smoke_cross_check(mock_root: Path, mock_datalist: Path, n_samples: int = 5) -> dict:
    """与真实跑一次官方 evaluate_mock.py CLI 的 CD_noisy 交叉核对。"""
    rels = read_datalist(mock_datalist)[:n_samples]

    # (a) 本脚本直接调用同一函数
    own_values = {}
    for rel in rels:
        noisy_path = mock_root / rel / "noisy.npy"
        clean_path = mock_root / rel / "clean.npy"
        own_values[rel] = compute_cd_noisy_one(noisy_path, clean_path)

    # (b) 真实跑一次官方 evaluate_mock.py CLI，用它自己的执行路径产出 CD_noisy
    #     需要一个 pred_dir（评测脚本硬性要求），拿 noisy 当 pred 占位——
    #     不影响 CD_noisy 的读数，CD_noisy 只依赖 noisy/gt 两个文件。
    smoke_datalist_path = STARTER_CODE / "datalist" / "_score_weight_smoke_5.txt"
    smoke_datalist_path.write_text("\n".join(rels) + "\n", encoding="utf-8")

    pred_dir = REPO_ROOT / "outputs" / "diagnostics" / VERSION / "_tmp_score_weight_smoke_pred"
    for rel in rels:
        src = mock_root / rel / "noisy.npy"
        dst = pred_dir / rel / "denoised.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, np.load(src))

    mesh_root = resolve("dataset/train")
    cmd = [
        sys.executable, str(STARTER_CODE / "evaluate_mock.py"),
        "--pred_dir", str(pred_dir),
        "--gt_dir", str(mock_root), "--noisy_dir", str(mock_root),
        "--mesh_dir", str(mesh_root),
        "--datalist", str(smoke_datalist_path.relative_to(STARTER_CODE)),
        "--workers", "1",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(STARTER_CODE))
    if result.returncode != 0:
        raise SystemExit(
            f"[FAIL] 官方 evaluate_mock.py CLI 交叉核对跑失败 (exit={result.returncode})\n"
            f"stdout(tail): {result.stdout[-1500:]}\nstderr(tail): {result.stderr[-1500:]}"
        )

    print("[SMOKE] 官方 CLI 输出:")
    print(result.stdout[-1500:])

    # 因为 pred=noisy，CD_pred 应该几乎等于 CD_noisy（自比较），metric_to_score(CD_pred=CD_noisy)
    # 应该给出接近 0 分（1 - CD_pred/CD_noisy ≈ 0）。这不是我们要验证的目标——
    # 目标是 own_values 与官方 CLI 内部逐样本计算出的 CD_noisy 是否一致，
    # 但官方 CLI 的 per-sample CD_noisy 不直接打印到 stdout（只打印汇总均值），
    # 所以改用"官方 CLI 汇总的 CD_noisy 均值" vs "own_values 的均值"做交叉核对——
    # 5 个样本子集下，均值层面的一致性足以验证调用路径正确。
    import re
    m = re.search(r"平均 CD_noisy:\s*([\d.]+)", result.stdout)
    if not m:
        raise SystemExit("[FAIL] 未能从官方 CLI 输出解析到 平均 CD_noisy")
    official_mean_cd_noisy = float(m.group(1))
    own_mean_cd_noisy = float(np.mean(list(own_values.values())))

    rel_diff = abs(official_mean_cd_noisy - own_mean_cd_noisy) / max(official_mean_cd_noisy, 1e-12)

    import shutil
    shutil.rmtree(pred_dir, ignore_errors=True)
    smoke_datalist_path.unlink(missing_ok=True)

    return {
        "n_samples": len(rels),
        "own_per_sample_cd_noisy": own_values,
        "own_mean_cd_noisy": own_mean_cd_noisy,
        "official_cli_mean_cd_noisy": official_mean_cd_noisy,
        "relative_diff": rel_diff,
        "pass_lt_1pct": rel_diff < 0.01,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--mock-root", default=DEFAULT_MOCK_ROOT)
    parser.add_argument("--mock-datalist", default=DEFAULT_MOCK_DATALIST)
    parser.add_argument("--smoke-check", action="store_true",
                         help="先跑 5 个样本与官方评测脚本交叉核对，通过才做全量预计算")
    parser.add_argument("--run-id", default="",
                         help="输出目录名。留空则用时间戳")
    parser.add_argument("--out", default="",
                         help="权重 json 输出路径。默认写在输出目录下的 cd_noisy_sidecar.json")
    args = parser.parse_args()

    train_root = resolve(args.train_root)
    mock_root = resolve(args.mock_root)
    mock_datalist = resolve(args.mock_datalist)
    datalist_path = resolve(args.datalist)
    rels = read_datalist(datalist_path)

    print(f"[INFO] script_version={SCRIPT_VERSION}")
    print(f"[INFO] datalist={datalist_path} (n={len(rels)})")
    print(f"[INFO] train_root={train_root}")

    smoke_result = None
    if args.smoke_check:
        print("\n[SMOKE] 5 样本 CD_noisy 与官方 evaluate_mock.py CLI 交叉核对...")
        smoke_result = smoke_cross_check(mock_root, mock_datalist, n_samples=5)
        print(f"  own_mean_cd_noisy={smoke_result['own_mean_cd_noisy']:.8f}")
        print(f"  official_cli_mean_cd_noisy={smoke_result['official_cli_mean_cd_noisy']:.8f}")
        print(f"  relative_diff={smoke_result['relative_diff']*100:.4f}%")
        if not smoke_result["pass_lt_1pct"]:
            raise SystemExit(
                f"[FAIL] smoke 交叉核对未过 <1% 门槛（relative_diff="
                f"{smoke_result['relative_diff']*100:.4f}%），停止全量预计算。"
            )
        print("  [SMOKE] PASS（<1%），继续全量预计算")

    print(f"\n[PHASE] 全量预计算 CD_noisy（{len(rels)} shapes，noisy.npy vs clean.npy）...")
    t0 = time.time()
    cd_noisy_map: dict[str, float] = {}
    missing = []
    for i, rel in enumerate(rels, 1):
        noisy_path = train_root / rel / "noisy.npy"
        clean_path = train_root / rel / "clean.npy"
        if not noisy_path.exists() or not clean_path.exists():
            missing.append(rel)
            continue
        cd_noisy_map[rel] = compute_cd_noisy_one(noisy_path, clean_path)
        if i % 200 == 0:
            print(f"  {i}/{len(rels)}", flush=True)
    elapsed = time.time() - t0

    if missing:
        raise SystemExit(f"[FAIL] {len(missing)} 个样本缺 noisy.npy/clean.npy: {missing[:5]}")

    values = np.array(list(cd_noisy_map.values()))
    median_cd_noisy = float(np.median(values))
    mean_cd_noisy = float(np.mean(values))

    print(f"\n[STATS] n={len(values)} median={median_cd_noisy:.8f} mean={mean_cd_noisy:.8f} "
          f"min={values.min():.8f} max={values.max():.8f}")
    print(f"[TIMING] {elapsed:.1f}s total, {elapsed/len(values)*1000:.1f}ms/shape")

    # 权重预计算：w_i = clip(median / cd_noisy_i, 0.5, 3.0)，E[w] 在全体上一次性算
    raw_w = {rel: float(np.clip(median_cd_noisy / v, 0.5, 3.0)) for rel, v in cd_noisy_map.items()}
    e_w = float(np.mean(list(raw_w.values())))
    w_norm = {rel: w / e_w for rel, w in raw_w.items()}

    w_norm_values = np.array(list(w_norm.values()))
    print(f"[W_NORM] E[w_raw]={e_w:.6f}  w_norm: min={w_norm_values.min():.4f} "
          f"median={np.median(w_norm_values):.4f} max={w_norm_values.max():.4f}")

    run_id = args.run_id or (
        time.strftime("%Y%m%d_%H%M%S") + "_a_final_score_weights"
    )
    out_dir = REPO_ROOT / "outputs" / "diagnostics" / VERSION / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecar_path = resolve(args.out) if args.out else (out_dir / "cd_noisy_sidecar.json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("w", encoding="utf-8") as f:
        json.dump({
            "cd_noisy": cd_noisy_map,
            "median_cd_noisy": median_cd_noisy,
            "e_w_raw": e_w,
            "w_norm": w_norm,
        }, f, indent=2, ensure_ascii=False)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/pipeline/04_precompute_score_weights.py",
        "script_version": SCRIPT_VERSION,
        "datalist": str(datalist_path),
        "n_shapes": len(values),
        "train_root": str(train_root),
        "cd_noisy_source_function": "starter_code.evaluate_mock.chamfer_distance (直接 import，同官方口径)",
        "median_cd_noisy": median_cd_noisy,
        "mean_cd_noisy": mean_cd_noisy,
        "e_w_raw": e_w,
        "w_norm_stats": {
            "min": float(w_norm_values.min()),
            "median": float(np.median(w_norm_values)),
            "max": float(w_norm_values.max()),
        },
        "precompute_time_sec": round(elapsed, 1),
        "sidecar_path": str(sidecar_path),
        "smoke_cross_check": smoke_result,
        "note": "权重只依据 CD 分母，不含 P2S 项；对已被 clamp 到 0/100 的样本是近似。",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] sidecar -> {sidecar_path}")
    print(f"[DONE] manifest -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
