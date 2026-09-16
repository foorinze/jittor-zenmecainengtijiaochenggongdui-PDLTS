#!/usr/bin/env python3
"""第三阶段：几何方向场后处理。

在第二阶段输出之上再施加一个固定的位移场。位移方向由点云自身的多尺度局部
几何算出，位移幅度固定为点间距的 2%。

机制：对每个点，在 5 个邻域尺度（k = 4/8/16/32/64）上各算两类局部向量——

    1. 邻域质心方向：(邻域重心 − 自身) / h，反映该点相对局部密度中心的偏移；
    2. 邻域斥力方向：−Σ u / (|u|² + eps)，其中 u 是归一化的邻居相对向量，
       反映该点被近邻挤压的方向。

5 个尺度 × 2 类 = 10 个基向量场。最终位移方向是这 10 个场的固定线性组合，
组合系数由岭回归在**训练集**的 52 个形状上拟合得到（见下方「系数来源」），
已冻结在本文件里，推理时不再拟合。

得到方向场后，把它整体归一化到 RMS = 0.02·h 再叠加：

    direction = Σ_m coefficient_m · (basis_m / basis_scale_m)
    displacement = h · direction · (0.02 / rms(direction))
    output = input + displacement

归一化这一步是必要的：岭回归给出的方向场量级依赖拟合数据的尺度，直接叠加
会让位移幅度随形状漂移。固定 RMS 剂量把「往哪走」与「走多远」解耦，只保留
方向信息，幅度由 h 决定，因此对点云的整体缩放是等变的。

系数来源与合规性：
    系数在训练集的 52 个形状上用岭回归拟合（13 类各 4 个，lambda=100，
    清单见 starter_code/datalist/controller_fit_shapes.txt）。拟合目标是官方复合
    评分的上升方向：CD 与 P2S 两项梯度各自用该样本自身的 noisy 误差归一化后
    相加，每项仅在当前仍有改善空间时计入。

    拟合过程本身在 scripts/pipeline/07a_fit_direction_controller.py，可独立重跑；
    该脚本的 --verify-frozen 会复算并与本文件的冻结值逐项比较。

    拟合阶段读取训练集的真值点云（dataset/a_final_train_full15k）与训练集网格
    （dataset/train，P2S 梯度需要点到三角面距离），与官方测试集无交集。
    **推理阶段（即本脚本）只读输入点云自身的坐标**，不读真值、网格、
    原始带噪点云或类别标签。

用法：
    python scripts/pipeline/07_apply_direction_controller.py \\
        --source outputs/predictions/a_final/repro_specialist_official/pred \\
        --out outputs/predictions/a_final/repro_final_official/pred

    从零重训时，上游权重与最终提交所用不同，应改用自己拟合出的系数：
        ... --model-json outputs/runs/a_final/repro_controller_fit/controller_model.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

SCRIPT_VERSION = "public"

# 邻域尺度。10 个基 = 5 个尺度 × {质心, 斥力} 两类。
K_VALUES = (4, 8, 16, 32, 64)
# 斥力项的软化半径，避免最近邻距离趋零时数值爆掉。
REPULSION_EPS = 0.25
# 位移幅度：RMS = DOSE_OVER_H × h，h 为该点云的中位最近邻间距。
DOSE_OVER_H = 0.02

# 冻结的岭回归系数（lambda=100，训练集 52 形状拟合）。顺序与 multiscale_bases
# 的输出一致：先 5 个质心尺度，再 5 个斥力尺度。
COEFFICIENTS = (
    0.0006431602218280801,
    0.0007693530545132081,
    7.591025305933538e-05,
    -0.0005880306536195402,
    0.000270848470719798,
    0.0006939779838470047,
    0.0006733555537263872,
    -0.00039149588958990196,
    -0.0012529249335484637,
    0.0010622119494412222,
)
# 各基在拟合集上的 RMS，用于把 10 个基归一化到同一量级后再做线性组合。
BASIS_SCALE = (
    0.2767319620672278,
    0.25544635124799386,
    0.28851715405256567,
    0.37121834445534124,
    0.5228678799666656,
    0.14283800455638448,
    0.092638924601768,
    0.06896060471614482,
    0.053333167062658886,
    0.0419805589017594,
)


def cloud_rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(np.square(value), axis=-1))))


def multiscale_bases(points: np.ndarray) -> tuple[np.ndarray, float]:
    """算 10 个基向量场，并返回该点云的中位最近邻间距 h。

    返回 (N, 10, 3) 与 h。只用点云自身坐标。
    """
    points = np.asarray(points, dtype=np.float64)
    n_points = len(points)
    distances, indices = cKDTree(points).query(
        points, k=max(K_VALUES) + 1, workers=-1
    )
    # 第 0 列是自身（距离 0），第 1 列才是最近邻
    h = float(np.median(distances[:, 1]))
    if h <= 0.0 or not np.isfinite(h):
        raise ValueError("点云间距退化，无法归一化")

    neighbor_indices = np.asarray(indices[:, 1:], dtype=np.int64)
    # u：邻居相对本点的位移，按 h 归一化 → 对整体缩放不变
    u = (points[neighbor_indices] - points[:, None, :]) / h
    u_sq = np.sum(np.square(u), axis=2, keepdims=True)
    repulsion = -u / (u_sq + REPULSION_EPS)

    # 前缀和让 5 个尺度共用一次 KNN，不必按 k 重复查询
    centroid_cumsum = np.cumsum(u, axis=1)
    repulsion_cumsum = np.cumsum(repulsion, axis=1)

    basis = []
    for k in K_VALUES:
        basis.append(centroid_cumsum[:, k - 1] / k)
    for k in K_VALUES:
        basis.append(repulsion_cumsum[:, k - 1] / k)

    result = np.stack(basis, axis=1)
    if result.shape != (n_points, 10, 3) or not np.isfinite(result).all():
        raise RuntimeError("基向量场非法")
    return result, h


def ridge_predict(
    basis: np.ndarray, coefficients: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """10 个基的固定线性组合，得到逐点方向。"""
    normalized = basis / scale[None, :, None]
    prediction = np.einsum("nmc,m->nc", normalized, coefficients)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("方向场出现非有限值")
    return prediction


def load_coefficients(model_json: str | None) -> tuple[np.ndarray, np.ndarray, str]:
    """取系数。默认用本文件里的冻结值；给了 --model-json 就用那次拟合的结果。

    冻结值对应最终提交所用的第二阶段权重。从零重训会得到略有差异的上游输出，
    对应的系数也会略有差异，此时应该用 07a 在自己权重上拟合出的 controller_model.json。
    """
    if not model_json:
        return (
            np.asarray(COEFFICIENTS, dtype=np.float64),
            np.asarray(BASIS_SCALE, dtype=np.float64),
            "frozen",
        )
    path = resolve(model_json)
    if not path.is_file():
        raise SystemExit(f"[FAIL] 系数文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    coefficients = np.asarray(payload["coefficients"], dtype=np.float64)
    scale = np.asarray(payload["basis_scale"], dtype=np.float64)
    if coefficients.shape != (10,) or scale.shape != (10,):
        raise SystemExit(f"[FAIL] 系数文件里应各有 10 个值: {path}")
    if not np.isfinite(coefficients).all() or not np.isfinite(scale).all():
        raise SystemExit(f"[FAIL] 系数文件含非有限值: {path}")
    if np.any(scale <= 0.0):
        raise SystemExit(f"[FAIL] basis_scale 必须为正: {path}")
    return coefficients, scale, str(path)


def apply_one(
    source: np.ndarray, coefficients: np.ndarray, scale: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """对一个点云施加位移。返回 (输出, h, 实际施加的 rms/h)。"""
    basis, h = multiscale_bases(source)
    direction = ridge_predict(basis, coefficients, scale)
    rms = cloud_rms(direction)
    if rms <= 0.0:
        raise ValueError("方向场全零")
    displacement = h * direction * (DOSE_OVER_H / rms)
    output = (source + displacement).astype(np.float32)
    return output, h, cloud_rms(displacement) / h


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                         help="第二阶段输出的 pred/ 目录")
    parser.add_argument("--out", required=True,
                         help="输出 pred/ 目录")
    parser.add_argument("--expect-samples", type=int, default=200,
                         help="期望样本数，不符则中止。0 = 不检查")
    parser.add_argument("--model-json", default="",
                         help="07a 拟合产出的 controller_model.json。"
                              "留空则用本文件里的冻结系数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = resolve(args.source)
    out_root = resolve(args.out)

    if not source_root.is_dir():
        raise SystemExit(f"[FAIL] 输入目录不存在: {source_root}")

    samples = sorted(
        str(p.parent.relative_to(source_root)).replace("\\", "/")
        for p in source_root.glob("*/*/*/denoised.npy")
    )
    if not samples:
        raise SystemExit(f"[FAIL] {source_root} 下没找到 denoised.npy")
    if args.expect_samples and len(samples) != args.expect_samples:
        raise SystemExit(
            f"[FAIL] 样本数 {len(samples)} != 期望 {args.expect_samples}，"
            f"中止以免产出不完整结果"
        )

    coefficients, scale, coef_source = load_coefficients(args.model_json)

    print(f"[方向场后处理] script_version={SCRIPT_VERSION}")
    print(f"  输入: {source_root}")
    print(f"  输出: {out_root}")
    print(f"  样本数: {len(samples)}")
    print(f"  剂量: RMS = {DOSE_OVER_H} × h")
    print(f"  系数: {coef_source}")

    t0 = time.time()
    rows = []
    max_dose_error = 0.0
    for i, sample in enumerate(samples, start=1):
        source = np.load(source_root / sample / "denoised.npy").astype(np.float64)
        output, h, applied = apply_one(source, coefficients, scale)
        if output.shape != source.shape or not np.isfinite(output).all():
            raise SystemExit(f"[FAIL] 输出非法: {sample}")

        dst = out_root / sample / "denoised.npy"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, output)

        max_dose_error = max(max_dose_error, abs(applied - DOSE_OVER_H))
        rows.append({"sample": sample, "h": h, "applied_rms_over_h": applied})
        if i % 20 == 0 or i == len(samples):
            print(f"  {i}/{len(samples)}", flush=True)

    elapsed = time.time() - t0

    # 剂量自检：实际施加的 RMS/h 必须等于设定值，否则归一化实现有问题
    if max_dose_error > 1e-10:
        raise SystemExit(
            f"[FAIL] 剂量偏差 {max_dose_error:.3e} 超过 1e-10，归一化实现有误"
        )

    summary = {
        "script": "scripts/pipeline/07_apply_direction_controller.py",
        "script_version": SCRIPT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source_root),
        "out": str(out_root),
        "n_samples": len(samples),
        "k_values": list(K_VALUES),
        "repulsion_eps": REPULSION_EPS,
        "dose_over_h": DOSE_OVER_H,
        "coefficients_source": coef_source,
        "coefficients": coefficients.tolist(),
        "basis_scale": scale.tolist(),
        "max_dose_error": max_dose_error,
        "wall_sec": round(elapsed, 2),
        "reads_ground_truth": False,
        "reads_mesh": False,
        "rows": rows,
    }
    summary_path = out_root.parent / "controller_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 写 manifest.json 供打包脚本校验。verdict 继承上游推理的结论：本步是逐点
    # 位移，不改点数也不涉及 patch 覆盖，不会引入新的漏点，因此完整性结论沿用
    # 上游。上游没有 manifest 时不臆造 green，写 unknown 让打包器按其规则处理。
    source_manifest_path = source_root.parent / "manifest.json"
    inherited_verdict = "unknown"
    source_summary = {}
    if source_manifest_path.is_file():
        try:
            with source_manifest_path.open("r", encoding="utf-8") as f:
                source_summary = (json.load(f) or {}).get("summary", {}) or {}
            inherited_verdict = source_summary.get("verdict", "unknown")
        except ValueError:
            inherited_verdict = "unknown"

    manifest = {
        "run_id": out_root.parent.name,
        "kind": "predict",
        "produced_by": "scripts/pipeline/07_apply_direction_controller.py",
        "source_predict_run": str(source_root.parent),
        "summary": {
            "n_samples": len(samples),
            "verdict": inherited_verdict,
            "verdict_source": (
                "inherited_from_upstream_predict"
                if source_manifest_path.is_file() else "upstream_manifest_absent"
            ),
            "n_missing_total": source_summary.get("n_missing_total", 0),
            "postprocess": "direction_controller",
            "point_count_changed": False,
        },
    }
    manifest_path = out_root.parent / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[完成] {elapsed:.1f}s，剂量最大偏差 {max_dose_error:.2e}")
    print(f"  继承上游 verdict: {inherited_verdict}")
    print(f"  摘要 -> {summary_path}")
    print(f"  manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
