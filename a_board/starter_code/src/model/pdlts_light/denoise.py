"""PDLTS Light 推理 / patch stitching。

推理必须保证输出点数 == 输入点数, 详见下方输入与输出约定.
原仓库对应: PD-LTS 原始实现 models/model_light/denoise.py:54-114.

禁用的原仓库路径 (会改点数):
    patch_denoise_validation, large_patch_denoise_*, remove_outliers, FPS downsample.

推理输入与输出约定:
    1. 归一化: predict_step 必须先对整云做 unit-sphere 归一化 (BBox center +
       L2-max scale), 再走 patch_denoise, 最后反归一化回原尺度. patch_denoise
       内部的 "patch 减 seed" 只是 patch-local 中心化, 不能代替全局归一化.
    2. stitching 距离归一化: 每个 patch 的 KNN 距离按 patch 最远邻居归一化,
       再取 argmin, 与原仓库 denoise.py:72 对齐.
    3. patch_size > N fallback: 不直接 ValueError, 改用 effective_patch_size =
       min(patch_size, N), 并在 README 标这是 smoke fallback.
    4. coverage 不完美的现实应对: FPS + KNN 在 N 接近 patch_size 时不保证全覆盖
       (2026-05-08 WSL 实测 N=128, K=6, patch=64 时漏点 1 个). 赛题硬约束要求
       输出点数 == 输入点数, 漏点 fail-loudly 会导致 0 分. 所以漏点用 noisy
       原值回填, 同时在 run logs 记 warn. 真实 50K + patch_size=1024 下预期
       零漏点或 << 0.1%, 这个回填是最后防线.
    5. K 用 math.ceil (不是 int/floor): 多 1 个 patch 的成本小, 覆盖率收益大.
"""

import math
import warnings
from typing import Dict, Tuple

import jittor as jt
import numpy as np

from ..vm import farthest_point_sampling
from .layer import safe_knn


# ---------------------------------------------------------------------------
# 归一化 (对齐 starter_code/src/data/augment.py:48 AugmentNormalizePC 的训练口径)
# ---------------------------------------------------------------------------
def normalize_unit_sphere(pc_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """把 (N, 3) 点云归一化到单位球内.

    与训练时 AugmentNormalizePC 口径一致:
        center = (max + min) / 2
        scale  = sqrt(max((pc - center)**2 . sum(axis=1)))
        out    = (pc - center) / scale

    Returns:
        normalized: (N, 3)
        center: (3,)
        scale: scalar

    反归一化: `pc * scale + center`.
    """
    assert pc_np.ndim == 2 and pc_np.shape[-1] == 3
    p_max = pc_np.max(axis=0)
    p_min = pc_np.min(axis=0)
    center = (p_max + p_min) / 2.0
    shifted = pc_np - center
    scale = float(np.sqrt((shifted ** 2).sum(axis=1).max()))
    if scale < 1e-12:
        # 退化: 全部点重合. 让 scale=1 避免除零; 此时 shifted 本身就是 0 向量.
        scale = 1.0
    return shifted / scale, center.astype(np.float32), float(scale)


def denormalize_unit_sphere(pc_np: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return pc_np * scale + center


# ---------------------------------------------------------------------------
# Patch-stitching 推理
# ---------------------------------------------------------------------------
@jt.no_grad()
def patch_denoise(
    network,
    pcl_noisy: jt.Var,
    patch_size: int = 1024,
    seed_k: int = 3,
    seed_k_alpha: int = 5,
    missing_fill: str = "noisy",
    return_coverage: bool = False,
    return_sidecars: bool = False,
):
    """PDLTS Light 单云推理 (已在归一化空间, 不含外层 normalize/denormalize).

    调用方 (如 PDLTSLight.predict_step) 必须先走 normalize_unit_sphere.

    Args:
        network: PDLTSLightNetwork 实例
        pcl_noisy: (N, 3) 已归一化的 noisy 点云
        patch_size: 每 patch 点数. 当 N < patch_size 时自动 fallback 到
            effective = min(patch_size, N) (smoke fallback).
        seed_k: 平均每点被多少 patch 覆盖; num_patches = ceil(seed_k * N / patch_size).
        seed_k_alpha: 分批送网络的倍数.
        missing_fill: 覆盖漏点的回填策略. 当前实现只支持 "noisy" (用输入 noisy 原值).
        return_coverage: True 时返回 (pcl_denoised, coverage_info), False 只返回
            pcl_denoised. coverage_info 是 dict, 含 n_missing / missing_ratio /
            K / effective_patch_size / seed_k. 供 PDLTSLightPredictSystem
            按样本汇总用.

    Returns:
        pcl_denoised: (N, 3), shape 严格等于输入 (赛题硬约束).
        (如果 return_coverage=True, 还返回 coverage_info dict.)
    """
    assert pcl_noisy.ndim == 2, f"pcl_noisy must be (N, 3), got {pcl_noisy.shape}"
    N, d = pcl_noisy.shape

    effective_patch_size = min(int(patch_size), int(N))
    # K 用 ceil 而不是 int/floor: 多 1 个 patch 的成本小, 覆盖率收益大.
    K = max(1, math.ceil(seed_k * N / effective_patch_size))

    # 1. FPS 采 K 个 seed
    pcl_noisy_b = pcl_noisy.unsqueeze(0)
    seed_pnts, _seed_idx = farthest_point_sampling(pcl_noisy_b, K)

    # 2. KNN: 每 seed 取 effective_patch_size 个邻居
    # 改用 safe_knn（带越界守卫的 jt.misc.knn 副本）。原版内核在
    # 副本）。此处 b*n=586 不能被 auto_parallel 的 256 整除，原版内核
    # b*n 不能被 256 整除时会向 idx 缓冲之后越界写入约 1.4MB，曾导致推理
    # （cudaErrorIllegalAddress 700）的根因；数值语义与原版完全一致。
    patch_dists, point_idxs = safe_knn(seed_pnts, pcl_noisy_b, effective_patch_size)
    patch_dists = patch_dists[0]            # (K, effective_patch_size)
    point_idxs = point_idxs[0]              # (K, effective_patch_size)

    # 3. gather patches
    flat_idx = point_idxs.reshape(-1)
    patches = pcl_noisy[flat_idx].reshape(K, effective_patch_size, d)

    # 4. 减 seed 中心化
    seed_pnts_1 = seed_pnts.squeeze(0).unsqueeze(1)       # (K, 1, 3)
    patches_centered = patches - seed_pnts_1.broadcast([K, effective_patch_size, d])

    # 5. 分批送网络
    batch_size = max(1, K // seed_k_alpha)
    denoised_chunks = []
    pull_w_chunks = []
    pull_vector_chunks = []
    for start in range(0, K, batch_size):
        end = min(start + batch_size, K)
        chunk = patches_centered[start:end]
        chunk_denoised, _ldj, _loss = network(chunk)
        denoised_chunks.append(chunk_denoised)
        if return_sidecars:
            pull_w = getattr(network, "_pull_w", None)
            pull_vector = getattr(network, "_pull_vector", None)
            if pull_w is None:
                pull_w = jt.zeros((chunk.shape[0], effective_patch_size, 1))
            if pull_vector is None:
                pull_vector = jt.zeros((chunk.shape[0], effective_patch_size, d))
            pull_w_chunks.append(pull_w)
            pull_vector_chunks.append(pull_vector)
    patches_denoised = jt.concat(denoised_chunks, dim=0)
    if return_sidecars:
        patches_pull_w = jt.concat(pull_w_chunks, dim=0)
        patches_pull_vector = jt.concat(pull_vector_chunks, dim=0)

    # 6. 加回 seed
    patches_denoised = patches_denoised + seed_pnts_1.broadcast([K, effective_patch_size, d])

    # 7. Stitching (numpy): 距离归一化 + argmin + coverage check + 漏点回填
    patch_dists_np = patch_dists.numpy()
    point_idxs_np = point_idxs.numpy().astype(np.int64)

    # 7a. 距离按 patch 最远邻居归一化 (对齐原仓库 denoise.py:72)
    denom = patch_dists_np[:, -1:] + 1e-8
    patch_dists_norm = patch_dists_np / denom

    # 7b. all_dists (K, N): 未覆盖位置留 inf
    all_dists = np.full((K, N), np.inf, dtype=np.float32)
    for k in range(K):
        all_dists[k, point_idxs_np[k]] = patch_dists_norm[k]

    # 7c. Coverage 检查 (不再 fail-loudly, 记 warn + 回填)
    covered = np.isfinite(all_dists).any(axis=0)
    n_missing = int((~covered).sum())
    if n_missing > 0:
        missing_ratio = n_missing / N
        # 记 warn 供调试; 正式预测路径靠 return_coverage=True 把 coverage_info 传给
        # 的 PDLTSLightPredictSystem, 由它汇总 n_missing 到 predict manifest
        # 并做绿/黄/红灯判定，并写入预测清单.
        if not return_coverage:
            warnings.warn(
                f"patch stitching coverage gap: {n_missing}/{N} points "
                f"({missing_ratio * 100:.2f}%) not covered by any patch. "
                f"K={K}, effective_patch_size={effective_patch_size}, "
                f"seed_k={seed_k}. Filling with '{missing_fill}' value.",
                RuntimeWarning,
            )

    # 7d. argmin 获取每个被覆盖点的 best patch (未覆盖的 argmin 会给 0 但后面会被漏点策略覆盖)
    best_patch_per_point = all_dists.argmin(axis=0)  # (N,)

    # 8. 按 best patch 挑 denoised 值
    patches_denoised_np = patches_denoised.numpy()    # (K, patch, 3)
    pcl_noisy_np = pcl_noisy.numpy()                  # (N, 3) 用于回填

    # 8a. 先用 noisy 初始化 (漏点的默认值)
    if missing_fill == "noisy":
        pcl_denoised_np = pcl_noisy_np.copy()
    else:
        raise ValueError(f"unsupported missing_fill: {missing_fill}")
    if return_sidecars:
        confidence_w_np = np.zeros((N,), dtype=np.float32)
        pull_vector_np = np.zeros((N, d), dtype=np.float32)
        patches_pull_w_np = patches_pull_w.numpy()
        patches_pull_vector_np = patches_pull_vector.numpy()

    # 8b. 覆盖点用 denoised 覆写
    for k in range(K):
        pts_in_k = np.where(covered & (best_patch_per_point == k))[0]
        if pts_in_k.size == 0:
            continue
        covered_idx = point_idxs_np[k]
        slot_map = {int(pi): slot for slot, pi in enumerate(covered_idx)}
        for p in pts_in_k:
            slot = slot_map[int(p)]
            pcl_denoised_np[p] = patches_denoised_np[k, slot]
            if return_sidecars:
                confidence_w_np[p] = float(patches_pull_w_np[k, slot, 0])
                pull_vector_np[p] = patches_pull_vector_np[k, slot]

    # 9. NaN / Inf guard: stitching 完成后不应有 NaN (noisy 输入也假定有限)
    if not np.isfinite(pcl_denoised_np).all():
        bad = (~np.isfinite(pcl_denoised_np)).sum()
        raise RuntimeError(
            f"patch_denoise produced {bad} non-finite values; "
            f"check ActNorm init / AffineCoupling log_scale range"
        )

    result = jt.array(pcl_denoised_np.astype(np.float32))
    sidecars: Dict[str, np.ndarray] = {}
    if return_sidecars:
        pull_norm_np = np.sqrt((pull_vector_np.astype(np.float64) ** 2).sum(axis=1)).astype(np.float32)
        sidecars = {
            "confidence_w": confidence_w_np.astype(np.float32),
            "pull_norm": pull_norm_np,
            "pull_vector": pull_vector_np.astype(np.float32),
        }
    if return_coverage:
        coverage_info = {
            "n_missing": int(n_missing),
            "missing_ratio": float(n_missing) / float(N),
            "K": int(K),
            "effective_patch_size": int(effective_patch_size),
            "seed_k": int(seed_k),
            "N": int(N),
        }
        if return_sidecars:
            return result, coverage_info, sidecars
        return result, coverage_info
    if return_sidecars:
        return result, sidecars
    return result


@jt.no_grad()
def denoise_full_cloud(
    network,
    pcl_noisy_np: np.ndarray,
    patch_size: int = 1024,
    seed_k: int = 3,
    seed_k_alpha: int = 5,
    return_coverage: bool = False,
    return_sidecars: bool = False,
):
    """完整推理入口: normalize -> patch_denoise -> denormalize.

    predict_step 内部调用这个. 参数语义和 patch_denoise 一致.

    Args:
        pcl_noisy_np: (N, 3) float32, 原始 noisy 坐标 (未归一化).
        return_coverage: True 时返回 (pcl_denoised_np, coverage_info). 供 预测落盘阶段
            PDLTSLightPredictSystem 按样本汇总 n_missing.

    Returns:
        pcl_denoised_np: (N, 3) float32, 在原始坐标系下的 denoised (shape 不变).
    """
    assert pcl_noisy_np.ndim == 2 and pcl_noisy_np.shape[-1] == 3
    normed_np, center, scale = normalize_unit_sphere(pcl_noisy_np.astype(np.float32))
    normed = jt.array(normed_np)
    pd_out = patch_denoise(
        network, normed,
        patch_size=patch_size, seed_k=seed_k, seed_k_alpha=seed_k_alpha,
        return_coverage=return_coverage,
        return_sidecars=return_sidecars,
    )
    if return_coverage and return_sidecars:
        denoised_normed, coverage_info, sidecars_normed = pd_out
    elif return_coverage:
        denoised_normed, coverage_info = pd_out
        sidecars_normed = None
    elif return_sidecars:
        denoised_normed, sidecars_normed = pd_out
    else:
        denoised_normed = pd_out
        sidecars_normed = None
    denoised_normed_np = denoised_normed.numpy()
    result = denormalize_unit_sphere(denoised_normed_np, center, scale).astype(np.float32)
    sidecars = None
    if sidecars_normed is not None:
        pull_vector = (sidecars_normed["pull_vector"].astype(np.float32) * float(scale)).astype(np.float32)
        pull_norm = (sidecars_normed["pull_norm"].astype(np.float32) * float(scale)).astype(np.float32)
        sidecars = {
            "confidence_w": sidecars_normed["confidence_w"].astype(np.float32),
            "pull_norm": pull_norm,
            "pull_vector": pull_vector,
        }
    if return_coverage and return_sidecars:
        return result, coverage_info, sidecars
    if return_coverage:
        return result, coverage_info
    if return_sidecars:
        return result, sidecars
    return result
