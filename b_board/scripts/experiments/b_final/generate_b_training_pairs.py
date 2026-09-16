#!/usr/bin/env python3
"""B 榜最终方案：从 train_b mesh 生成训练数据和健康样本清单。

复用同目录 mesh_sampling（网格采样）工具，为 train_b 全量 19699 样本生成点云训练数据。
（原写 19698，系 wc -l 对无末换行 datalist 少数 1；真值 grep -c . = 19699）
噪声参数使用 B 榜训练数据标定范围。

用法：
  cd b_board
  python scripts/experiments/b_final/generate_b_training_pairs.py \\
      --dataset-dir dataset/train_b \\
      --datalist starter_code/datalist/train_b.txt \\
      --output-datalist starter_code/datalist/train_b_generated.txt \\
      --num-workers 8
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# 导入同目录采样工具
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mesh_sampling import (
    load_obj_mesh,
    normalize_bbox_unit_sphere,
    sample_surface,
    add_laplace_noise,
    inspect_mesh,
)

# B 榜训练数据标定噪声范围
CALIBRATED_SIGMA_MIN = 0.0075
CALIBRATED_SIGMA_MAX = 0.0161

# 错误逐条打印的上限，超过后只计数，避免整片失败时刷爆日志
MAX_ERROR_LINES = 20


def process_one_sample(args_tuple):
    """处理单个样本：采样 → 归一化 → 加噪声。"""
    item, dataset_root, points, sigma_min, sigma_max, seed_base, seed_mode = args_tuple
    if seed_mode == "stable_sha256":
        item_seed = int(hashlib.sha256(item.encode("utf-8")).hexdigest()[:8], 16)
    else:
        # 保留 2026-08-11 首次生成训练对时的旧口径，仅用于来源核查。
        item_seed = hash(item) % (2**31)
    rng = np.random.default_rng(seed_base + item_seed % (2**31))

    mesh_path = dataset_root / item / "models" / "model_normalized.obj"
    try:
        # 读取 mesh
        vertices, faces = load_obj_mesh(str(mesh_path))

# 健康检查（复用数据健康检查闸门）
        health = inspect_mesh(vertices, faces)
        if health.is_degenerate or health.is_outlier_dominated:
            return {"item": item, "status": "rejected_health"}

        # 采样
        sampled = sample_surface(vertices, faces, points, rng)

        # 归一化
        clean, center, scale = normalize_bbox_unit_sphere(sampled)

        # 加噪声
        sigma = float(rng.uniform(sigma_min, sigma_max))
        noisy = add_laplace_noise(clean, sigma, rng)

        # 保存
        sample_dir = dataset_root / item
        np.save(sample_dir / "clean.npy", clean.astype(np.float32))
        np.save(sample_dir / "noisy.npy", noisy.astype(np.float32))
        (sample_dir / "norm.json").write_text(
            json.dumps({
                "type": "mesh_bbox_unit_sphere",
                "center": [float(v) for v in center],
                "scale": float(scale),
                "sigma": sigma,
            })
        )

        return {"item": item, "status": "success", "sigma": sigma}

    except Exception as exc:
        return {"item": item, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="dataset/train_b")
    parser.add_argument("--datalist", default="starter_code/datalist/train_b.txt")
    parser.add_argument(
        "--output-datalist",
        default="starter_code/datalist/train_b_generated.txt",
        help="按官方清单原顺序写入健康且生成成功的样本",
    )
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--sigma-min", type=float, default=CALIBRATED_SIGMA_MIN)
    parser.add_argument("--sigma-max", type=float, default=CALIBRATED_SIGMA_MAX)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument(
        "--seed-mode",
        choices=["stable_sha256", "legacy_python_hash"],
        default="stable_sha256",
        help="默认使用跨进程稳定的 SHA-256 派生 seed；legacy 仅用于来源核查",
    )
    parser.add_argument(
        "--expected-success",
        type=int,
        default=19528,
        help="健康样本数量闸门；设为 0 可关闭",
    )
    args = parser.parse_args()

    # 读取 datalist
    with open(args.datalist, encoding="utf-8") as f:
        items = [line.strip() for line in f if line.strip()]

    print(f"[INFO] 开始生成 {len(items)} 个样本的训练数据")
    print(f"[INFO] 采样点数: {args.points}, 噪声范围: [{args.sigma_min:.4f}, {args.sigma_max:.4f}]")
    print(f"[INFO] 并行进程数: {args.num_workers}")
    print(f"[INFO] seed_mode: {args.seed_mode}, seed: {args.seed}")
    if args.seed_mode == "legacy_python_hash" and "PYTHONHASHSEED" not in os.environ:
        print("[WARN] legacy_python_hash 未设置 PYTHONHASHSEED，跨运行不可复现")

    dataset_root = Path(args.dataset_dir)
    tasks = [
        (
            item,
            dataset_root,
            args.points,
            args.sigma_min,
            args.sigma_max,
            args.seed,
            args.seed_mode,
        )
        for item in items
    ]

    successful_items = set()
    rejected_count = 0
    error_count = 0
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_one_sample, task): task[0] for task in tasks}

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["status"] == "success":
                successful_items.add(result["item"])
            elif result["status"].startswith("rejected"):
                rejected_count += 1
            else:
                error_count += 1
                if error_count <= MAX_ERROR_LINES:
                    print(f"[ERROR] {result['item']}: {result.get('error', 'unknown')}")

            if i % 100 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(items) - i) / rate / 60
                print(f"[PROGRESS] {i}/{len(items)} ({100*i/len(items):.1f}%), "
                      f"{rate:.1f} samples/s, ETA {eta:.1f} min")

    generated_items = [item for item in items if item in successful_items]
    output_datalist = Path(args.output_datalist)
    output_datalist.parent.mkdir(parents=True, exist_ok=True)
    output_datalist.write_text(
        "\n".join(generated_items) + ("\n" if generated_items else ""),
        encoding="utf-8",
    )

    elapsed = time.time() - start_time
    print(f"\n[DONE] 完成 {len(items)} 个样本，耗时 {elapsed/60:.1f} 分钟")
    print(f"  成功: {len(generated_items)}")
    print(f"  健康闸剔除: {rejected_count}")
    print(f"  错误: {error_count}")
    print(f"  健康样本清单: {output_datalist}")

    if error_count:
        print("[FAIL] 存在读取或生成错误；请修复后重跑，避免静默改变训练集")
        return 2
    if args.expected_success and len(generated_items) != args.expected_success:
        print(
            f"[FAIL] 健康样本数 {len(generated_items)} != "
            f"expected {args.expected_success}"
        )
        return 3
    print("[PASS] 训练对和健康样本清单生成完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
