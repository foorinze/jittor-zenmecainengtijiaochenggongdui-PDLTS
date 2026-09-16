"""整云拼接后的候选点生成与选择工具。

包含 NumPy 候选点生成和 top-N 选择，以及用于梯度测试的 Jittor 评分器。
不改变局部块内的可逆网络或拼接算法；由配置开关控制，默认关闭。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jittor as jt
from jittor import nn
import numpy as np

from .layer import safe_knn

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected in the project env.
    cKDTree = None


@dataclass
class PostSelectConfig:
    postselect_mode: str = "off"
    candidate_generator_mode: str = "off"
    selector_mode: str = "identity"
    refine_mode: str = "off"
    candidate_ratio: float = 1.5
    candidate_knn_k: int = 8
    candidate_surface_reject: bool = False
    candidate_reject_k: int = 8
    candidate_reject_spacing_mult: float = 2.5
    candidate_reject_abs_threshold: float = 0.0
    replace_ratio: float = 0.02
    replace_drop_spacing_k: int = 8
    fps_exact_limit: int = 4096
    seed: int = 0


def _as_cloud(name: str, cloud: np.ndarray) -> np.ndarray:
    arr = np.asarray(cloud, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be (N, 3), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return arr


def _target_count(n_points: int, ratio: float) -> int:
    if ratio < 1.0:
        raise ValueError(f"candidate_ratio must be >= 1.0, got {ratio}")
    return max(int(n_points), int(np.ceil(float(n_points) * float(ratio))))


def _principal_sorted_midpoints(base: np.ndarray, extra_count: int, seed: int) -> np.ndarray:
    """Generate cheap geometry-only midpoint candidates without an NxN KNN matrix."""
    n_points = int(base.shape[0])
    if extra_count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    if n_points < 2:
        raise ValueError("need at least two base points to generate midpoint candidates")

    centered = base - base.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(float(n_points), 1.0)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    proj = base @ axis
    order = np.argsort(proj, kind="mergesort")
    if n_points > 1:
        order = np.roll(order, int(seed) % n_points)

    pair_pos = np.arange(extra_count, dtype=np.int64) % (n_points - 1)
    idx_a = order[pair_pos]
    idx_b = order[pair_pos + 1]
    return (0.5 * (base[idx_a] + base[idx_b])).astype(np.float32)


def _query_tree(tree, points: np.ndarray, k: int):
    """Compatibility wrapper for scipy versions with/without workers=."""
    try:
        return tree.query(points, k=k, workers=1)
    except TypeError:
        return tree.query(points, k=k)


def _require_ckdtree() -> None:
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required for local_knn_midpoint")


def _local_knn_midpoints(base: np.ndarray, extra_count: int, seed: int, knn_k: int) -> np.ndarray:
    """Generate midpoint candidates only from true 3D local neighbors."""
    _require_ckdtree()
    n_points = int(base.shape[0])
    if extra_count <= 0:
        return np.empty((0, 3), dtype=np.float32)
    if n_points < 2:
        raise ValueError("need at least two base points to generate local midpoint candidates")

    k_eff = max(1, min(int(knn_k), n_points - 1))
    tree = cKDTree(base.astype(np.float64))
    _, nn_idx = _query_tree(tree, base.astype(np.float64), k=k_eff + 1)
    nn_idx = np.asarray(nn_idx, dtype=np.int64)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    neighbors = nn_idx[:, 1:k_eff + 1]

    rng = np.random.default_rng(int(seed))
    replace = extra_count > n_points
    anchors = rng.choice(n_points, size=extra_count, replace=replace).astype(np.int64)
    neighbor_slot = rng.integers(0, k_eff, size=extra_count, endpoint=False)
    partners = neighbors[anchors, neighbor_slot]
    return (0.5 * (base[anchors] + base[partners])).astype(np.float32)


def _base_spacing(base: np.ndarray, spacing_k: int) -> np.ndarray:
    _require_ckdtree()
    n_points = int(base.shape[0])
    if n_points < 2:
        return np.ones((n_points,), dtype=np.float32)
    k_eff = max(1, min(int(spacing_k), n_points - 1))
    tree = cKDTree(base.astype(np.float64))
    dists, _ = _query_tree(tree, base.astype(np.float64), k=k_eff + 1)
    dists = np.asarray(dists, dtype=np.float32)
    if dists.ndim == 1:
        return np.maximum(dists, 1e-12).astype(np.float32)
    return np.maximum(dists[:, -1], 1e-12).astype(np.float32)


def _surface_reject_extra(
    base: np.ndarray,
    extra: np.ndarray,
    enabled: bool,
    spacing_k: int,
    spacing_mult: float,
    abs_threshold: float,
) -> Tuple[np.ndarray, Dict]:
    """Reject extra candidates that are far from the base surface proxy."""
    info = {
        "surface_reject_enabled": bool(enabled),
        "surface_reject_spacing_k": int(spacing_k),
        "surface_reject_spacing_mult": float(spacing_mult),
        "surface_reject_abs_threshold": float(abs_threshold),
        "extra_before_reject": int(extra.shape[0]),
        "extra_after_reject": int(extra.shape[0]),
        "extra_reject_rate": 0.0,
        "extra_nearest_base_dist_p95": 0.0,
    }
    if (not enabled) or extra.shape[0] == 0:
        return extra, info

    _require_ckdtree()
    tree = cKDTree(base.astype(np.float64))
    dists, nearest = _query_tree(tree, extra.astype(np.float64), k=1)
    dists = np.asarray(dists, dtype=np.float32)
    nearest = np.asarray(nearest, dtype=np.int64)
    spacing = _base_spacing(base, spacing_k=spacing_k)
    threshold = spacing[nearest] * float(spacing_mult)
    if float(abs_threshold) > 0.0:
        threshold = np.minimum(threshold, float(abs_threshold))
    keep = dists <= threshold
    filtered = extra[keep].astype(np.float32)
    info.update({
        "extra_after_reject": int(filtered.shape[0]),
        "extra_reject_rate": float(1.0 - (float(filtered.shape[0]) / max(float(extra.shape[0]), 1.0))),
        "extra_nearest_base_dist_p95": float(np.percentile(dists, 95)) if dists.size else 0.0,
    })
    return filtered, info


def generate_candidates(
    base: np.ndarray,
    noisy: Optional[np.ndarray] = None,
    mode: str = "off",
    ratio: float = 1.5,
    seed: int = 0,
    knn_k: int = 8,
    surface_reject: bool = False,
    reject_k: int = 8,
    reject_spacing_mult: float = 2.5,
    reject_abs_threshold: float = 0.0,
) -> Tuple[np.ndarray, Dict]:
    """Return candidates with the original base cloud as the first N points."""
    base = _as_cloud("base", base)
    if noisy is not None:
        _as_cloud("noisy", noisy)

    n_points = int(base.shape[0])
    if mode in ("off", "identity", "base"):
        return base.copy(), {
            "candidate_mode": mode,
            "num_base": n_points,
            "num_candidates": n_points,
            "num_extra": 0,
            "base_prefix": True,
        }
    if mode not in ("geometric", "pca_midpoint", "knn_midpoint", "local_knn_midpoint"):
        raise ValueError(f"unsupported candidate_generator_mode: {mode}")

    target = _target_count(n_points, ratio)
    extra_count = target - n_points
    if mode in ("knn_midpoint", "local_knn_midpoint"):
        extra = _local_knn_midpoints(base, extra_count, seed=seed, knn_k=knn_k)
        generator_impl = "local_knn_midpoint"
    else:
        extra = _principal_sorted_midpoints(base, extra_count, seed=seed)
        generator_impl = "pca_midpoint"
    extra, reject_info = _surface_reject_extra(
        base,
        extra,
        enabled=surface_reject,
        spacing_k=reject_k,
        spacing_mult=reject_spacing_mult,
        abs_threshold=reject_abs_threshold,
    )
    candidates = np.concatenate([base, extra], axis=0).astype(np.float32)
    info = {
        "candidate_mode": mode,
        "candidate_generator_impl": generator_impl,
        "num_base": n_points,
        "num_candidates": int(candidates.shape[0]),
        "num_extra": int(extra.shape[0]),
        "base_prefix": True,
        "candidate_knn_k": int(knn_k),
    }
    info.update(reject_info)
    return candidates, info


def identity_scores(num_candidates: int, keep_n: int) -> np.ndarray:
    if num_candidates < keep_n:
        raise ValueError(f"num_candidates={num_candidates} < keep_n={keep_n}")
    scores = np.full((num_candidates,), -1.0, dtype=np.float32)
    # Strictly decreasing scores keep the first keep_n points in their original order.
    scores[:keep_n] = 1.0 - (np.arange(keep_n, dtype=np.float32) * 1e-7)
    return scores


def full_cloud_select(
    candidates: np.ndarray,
    scores: np.ndarray,
    keep_n: int,
    return_index: bool = False,
):
    candidates = _as_cloud("candidates", candidates)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if candidates.shape[0] != scores.shape[0]:
        raise ValueError(
            f"scores length {scores.shape[0]} != candidates length {candidates.shape[0]}"
        )
    if keep_n <= 0:
        raise ValueError(f"keep_n must be positive, got {keep_n}")
    if candidates.shape[0] < keep_n:
        raise ValueError(f"candidate count {candidates.shape[0]} < keep_n {keep_n}")

    part = np.argpartition(-scores, keep_n - 1)[:keep_n]
    idx = part[np.argsort(-scores[part], kind="mergesort")]
    if np.unique(idx).shape[0] != keep_n:
        raise RuntimeError("hard top-N produced duplicate indices")
    selected = candidates[idx].astype(np.float32)
    if return_index:
        return selected, idx.astype(np.int64)
    return selected


def deterministic_fps_indices(
    candidates: np.ndarray,
    keep_n: int,
    seed: int = 0,
    exact_limit: int = 4096,
) -> Tuple[np.ndarray, str]:
    """Deterministic fixed selector for candidate-only control.

    For small smoke inputs this runs exact greedy FPS. For full-cloud P1a
    (75k -> 50k), exact O(MN) FPS is too expensive, so it falls back to a
    deterministic PCA-stratified spread selector. The fallback is still a fixed
    non-learned control and keeps unique candidate indices.
    """
    candidates = _as_cloud("candidates", candidates)
    num_candidates = int(candidates.shape[0])
    if keep_n <= 0:
        raise ValueError(f"keep_n must be positive, got {keep_n}")
    if num_candidates < keep_n:
        raise ValueError(f"candidate count {num_candidates} < keep_n {keep_n}")
    if num_candidates == keep_n:
        return np.arange(num_candidates, dtype=np.int64), "identity_all"

    if num_candidates <= int(exact_limit):
        selected = np.empty((keep_n,), dtype=np.int64)
        start = int(seed) % num_candidates
        selected[0] = start
        diff = candidates - candidates[start:start + 1]
        min_dist = (diff * diff).sum(axis=1)
        min_dist[start] = -1.0
        for i in range(1, keep_n):
            idx = int(np.argmax(min_dist))
            selected[i] = idx
            diff = candidates - candidates[idx:idx + 1]
            dist = (diff * diff).sum(axis=1)
            min_dist = np.minimum(min_dist, dist)
            min_dist[selected[:i + 1]] = -1.0
        return selected, "exact_greedy"

    centered = candidates - candidates.mean(axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(float(num_candidates), 1.0)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
        axis = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    order = np.argsort(candidates @ axis, kind="mergesort")
    positions = np.linspace(0, num_candidates - 1, keep_n, dtype=np.int64)
    idx = order[positions]
    if np.unique(idx).shape[0] != keep_n:
        # Extremely defensive fallback for pathological rounding collisions.
        idx = order[:keep_n]
    return idx.astype(np.int64), "pca_stratified_large"


def base_preserve_replace_indices(
    candidates: np.ndarray,
    keep_n: int,
    replace_ratio: float = 0.02,
    seed: int = 0,
    exact_limit: int = 4096,
    drop_spacing_k: int = 8,
) -> Tuple[np.ndarray, Dict]:
    """Select a near-identity subset by replacing only dense base slots.

    The function is intentionally zero-leakage: both the base drop order and
    extra fill order are computed only from candidate/base geometry.
    """
    candidates = _as_cloud("candidates", candidates)
    keep_n = int(keep_n)
    if keep_n <= 0 or keep_n > candidates.shape[0]:
        raise ValueError(f"invalid keep_n={keep_n} for candidates={candidates.shape}")
    base = candidates[:keep_n]
    extra = candidates[keep_n:]
    requested = int(round(float(keep_n) * float(replace_ratio)))
    replace_count = max(0, min(requested, int(extra.shape[0]), keep_n))
    if replace_count <= 0:
        idx = np.arange(keep_n, dtype=np.int64)
        return idx, {
            "selector_mode": "base_preserve_replace",
            "replace_ratio": float(replace_ratio),
            "replace_count_requested": int(requested),
            "replace_count_actual": 0,
            "drop_mode": "none",
            "extra_select_mode": "none",
            "selected_from_base": keep_n,
        }

    spacing = _base_spacing(base, spacing_k=int(drop_spacing_k))
    drop_idx = np.argsort(spacing, kind="mergesort")[:replace_count]
    keep_mask = np.ones((keep_n,), dtype=bool)
    keep_mask[drop_idx] = False
    keep_base_idx = np.nonzero(keep_mask)[0].astype(np.int64)

    extra_idx, backend = deterministic_fps_indices(
        extra,
        keep_n=replace_count,
        seed=int(seed),
        exact_limit=int(exact_limit),
    )
    idx = np.concatenate([keep_base_idx, keep_n + extra_idx.astype(np.int64)], axis=0)
    return idx.astype(np.int64), {
        "selector_mode": "base_preserve_replace",
        "replace_ratio": float(replace_ratio),
        "replace_count_requested": int(requested),
        "replace_count_actual": int(replace_count),
        "drop_mode": "densest_base_by_knn_spacing",
        "drop_spacing_k": int(drop_spacing_k),
        "extra_select_mode": "deterministic_fps",
        "fps_backend": backend,
        "selected_from_base": int(keep_n - replace_count),
    }


def base_preserve_learned_replace_indices(
    candidates: np.ndarray,
    keep_n: int,
    extra_scores: np.ndarray,
    replace_ratio: float = 0.02,
    drop_scores: Optional[np.ndarray] = None,
    drop_spacing_k: int = 8,
) -> Tuple[np.ndarray, Dict]:
    """Replace K dense base slots with the K highest-scored extras.

    `extra_scores` and optional `drop_scores` are inference-visible selector
    outputs. Clean-derived labels must never be passed here.
    """
    candidates = _as_cloud("candidates", candidates)
    keep_n = int(keep_n)
    if keep_n <= 0 or keep_n > candidates.shape[0]:
        raise ValueError(f"invalid keep_n={keep_n} for candidates={candidates.shape}")
    extra = candidates[keep_n:]
    requested = int(round(float(keep_n) * float(replace_ratio)))
    replace_count = max(0, min(requested, int(extra.shape[0]), keep_n))
    extra_scores = np.asarray(extra_scores, dtype=np.float32).reshape(-1)
    if extra_scores.shape[0] != extra.shape[0]:
        raise ValueError(
            f"extra_scores length {extra_scores.shape[0]} != num_extra {extra.shape[0]}"
        )
    if replace_count <= 0:
        idx = np.arange(keep_n, dtype=np.int64)
        return idx, {
            "selector_mode": "learned_replace",
            "replace_ratio": float(replace_ratio),
            "replace_count_requested": int(requested),
            "replace_count_actual": 0,
            "drop_mode": "none",
            "extra_select_mode": "none",
            "selected_from_base": keep_n,
        }

    if drop_scores is None:
        spacing = _base_spacing(candidates[:keep_n], spacing_k=int(drop_spacing_k))
        drop_idx = np.argsort(spacing, kind="mergesort")[:replace_count]
        drop_mode = "densest_base_by_knn_spacing"
    else:
        drop_scores = np.asarray(drop_scores, dtype=np.float32).reshape(-1)
        if drop_scores.shape[0] != keep_n:
            raise ValueError(f"drop_scores length {drop_scores.shape[0]} != keep_n {keep_n}")
        # Higher drop score means the base slot is more replaceable.
        drop_idx = np.argsort(-drop_scores, kind="mergesort")[:replace_count]
        drop_mode = "learned_drop_scores"
    keep_mask = np.ones((keep_n,), dtype=bool)
    keep_mask[drop_idx] = False
    keep_base_idx = np.nonzero(keep_mask)[0].astype(np.int64)

    extra_idx = np.argsort(-extra_scores, kind="mergesort")[:replace_count].astype(np.int64)
    idx = np.concatenate([keep_base_idx, keep_n + extra_idx], axis=0)
    return idx.astype(np.int64), {
        "selector_mode": "learned_replace",
        "replace_ratio": float(replace_ratio),
        "replace_count_requested": int(requested),
        "replace_count_actual": int(replace_count),
        "drop_mode": drop_mode,
        "drop_spacing_k": int(drop_spacing_k),
        "extra_select_mode": "learned_topk_extra",
        "selected_from_base": int(keep_n - replace_count),
        "extra_score_mean": float(extra_scores.mean()) if extra_scores.size else 0.0,
        "extra_score_std": float(extra_scores.std()) if extra_scores.size else 0.0,
    }


def project_local_to_base_tangent(
    selected: np.ndarray,
    base: np.ndarray,
    k: int = 16,
) -> Tuple[np.ndarray, Dict]:
    """Project selected points onto local base-only tangent planes.

    This is a deterministic zero-train surface probe. It uses only the frozen
    base cloud, never clean GT.
    """
    selected = _as_cloud("selected", selected)
    base = _as_cloud("base", base)
    if selected.shape[0] == 0:
        return selected.copy(), {"refine_mode": "project_local", "projected_points": 0}
    _require_ckdtree()
    k_eff = max(3, min(int(k), int(base.shape[0])))
    tree = cKDTree(base.astype(np.float64))
    _dists, nn_idx = _query_tree(tree, selected.astype(np.float64), k=k_eff)
    nn_idx = np.asarray(nn_idx)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]
    refined = selected.astype(np.float64).copy()
    moved = []
    for i in range(selected.shape[0]):
        neigh = base[nn_idx[i]].astype(np.float64)
        center = neigh.mean(axis=0)
        centered = neigh - center
        try:
            _vals, vecs = np.linalg.eigh(centered.T @ centered)
            normal = vecs[:, 0]
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        delta = float((refined[i] - center) @ normal)
        refined[i] = refined[i] - delta * normal
        moved.append(abs(delta))
    moved_np = np.asarray(moved, dtype=np.float32)
    return refined.astype(np.float32), {
        "refine_mode": "project_local",
        "refine_basis": "base_only_local_pca",
        "refine_k": int(k_eff),
        "projected_points": int(selected.shape[0]),
        "project_abs_delta_mean": float(moved_np.mean()) if moved_np.size else 0.0,
        "project_abs_delta_p95": float(np.percentile(moved_np, 95)) if moved_np.size else 0.0,
    }


def apply_postselect(
    base: np.ndarray,
    noisy: Optional[np.ndarray],
    config: PostSelectConfig,
) -> Tuple[np.ndarray, Dict]:
    """Apply postselect to a stitched full cloud.

    The off mode is an exact identity path and leaves the input unchanged.
    """
    base = _as_cloud("base", base)
    keep_n = int(base.shape[0])
    if config.postselect_mode == "off":
        return base.copy(), {
            "enabled": False,
            "postselect_mode": "off",
            "selector_mode": "off",
            "num_candidates": keep_n,
            "keep_n": keep_n,
            "coverage_reference": "base_stitching",
            "realized_coverage_recomputed": False,
        }
    if config.postselect_mode not in ("geometric", "candidate_select"):
        raise ValueError(f"unsupported postselect_mode: {config.postselect_mode}")

    candidates, cand_info = generate_candidates(
        base,
        noisy=noisy,
        mode=config.candidate_generator_mode,
        ratio=config.candidate_ratio,
        seed=config.seed,
        knn_k=config.candidate_knn_k,
        surface_reject=config.candidate_surface_reject,
        reject_k=config.candidate_reject_k,
        reject_spacing_mult=config.candidate_reject_spacing_mult,
        reject_abs_threshold=config.candidate_reject_abs_threshold,
    )
    if config.selector_mode in ("off", "identity", "base_first"):
        scores = identity_scores(candidates.shape[0], keep_n)
    elif config.selector_mode in ("fps", "deterministic_fps"):
        idx, backend = deterministic_fps_indices(candidates, keep_n, seed=config.seed)
        selected = candidates[idx].astype(np.float32)
        info = dict(cand_info)
        info.update({
            "enabled": True,
            "postselect_mode": config.postselect_mode,
            "selector_mode": config.selector_mode,
            "keep_n": keep_n,
            "selected_unique": int(np.unique(idx).shape[0]),
            "selected_from_base": int((idx < keep_n).sum()),
            "coverage_reference": "preselect_base_stitching",
            "realized_coverage_recomputed": False,
            "fps_backend": backend,
        })
        if config.refine_mode == "project_local":
            selected, refine_info = project_local_to_base_tangent(selected, base)
            info.update(refine_info)
        elif config.refine_mode not in ("off", "identity"):
            raise ValueError(f"unsupported refine_mode: {config.refine_mode}")
        else:
            info["refine_mode"] = "off"
        return selected, info
    elif config.selector_mode == "base_preserve_replace":
        idx, sel_info = base_preserve_replace_indices(
            candidates,
            keep_n=keep_n,
            replace_ratio=config.replace_ratio,
            seed=config.seed,
            exact_limit=config.fps_exact_limit,
            drop_spacing_k=config.replace_drop_spacing_k,
        )
        selected = candidates[idx].astype(np.float32)
        info = dict(cand_info)
        info.update(sel_info)
        info.update({
            "enabled": True,
            "postselect_mode": config.postselect_mode,
            "selector_mode": config.selector_mode,
            "keep_n": keep_n,
            "selected_unique": int(np.unique(idx).shape[0]),
            "coverage_reference": "preselect_base_stitching",
            "realized_coverage_recomputed": False,
        })
        if config.refine_mode == "project_local":
            selected, refine_info = project_local_to_base_tangent(selected, base)
            info.update(refine_info)
        elif config.refine_mode not in ("off", "identity"):
            raise ValueError(f"unsupported refine_mode: {config.refine_mode}")
        else:
            info["refine_mode"] = "off"
        return selected, info
    elif config.selector_mode == "x_desc":
        scores = candidates[:, 0].astype(np.float32)
    elif config.selector_mode == "random":
        rng = np.random.default_rng(int(config.seed))
        scores = rng.standard_normal(candidates.shape[0]).astype(np.float32)
    elif config.selector_mode == "learned":
        raise NotImplementedError("learned selector inference is reserved for P1")
    else:
        raise ValueError(f"unsupported selector_mode: {config.selector_mode}")

    selected, idx = full_cloud_select(candidates, scores, keep_n, return_index=True)
    info = dict(cand_info)
    info.update({
        "enabled": True,
        "postselect_mode": config.postselect_mode,
        "selector_mode": config.selector_mode,
        "keep_n": keep_n,
        "selected_unique": int(np.unique(idx).shape[0]),
        "selected_from_base": int((idx < keep_n).sum()),
        "coverage_reference": "preselect_base_stitching",
        "realized_coverage_recomputed": False,
    })
    if config.refine_mode == "project_local":
        selected, refine_info = project_local_to_base_tangent(selected, base)
        info.update(refine_info)
    elif config.refine_mode not in ("off", "identity"):
        raise ValueError(f"unsupported refine_mode: {config.refine_mode}")
    else:
        info["refine_mode"] = "off"
    return selected, info


def candidate_features_jt(candidates: jt.Var) -> jt.Var:
    """Build blind candidate features for gradient smoke tests."""
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates must be (M, 3), got {candidates.shape}")
    sq_norm = (candidates * candidates).sum(dim=-1, keepdims=True)
    abs_xyz = jt.sqrt(candidates * candidates + 1e-12)
    return jt.concat([candidates, sq_norm, abs_xyz], dim=-1)


class BlindCandidateScorer(nn.Module):
    """Tiny blind scorer used by selector experiments.

    Inputs must be inference-visible features only; clean-derived features are
    intentionally not part of this module contract.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 32,
        initial_score_bias: Optional[float] = None,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        final = nn.Linear(int(hidden_channels), 1)
        if initial_score_bias is not None:
            final.bias = jt.array(
                np.full(final.bias.shape, float(initial_score_bias), dtype=np.float32)
            )
        self.net = nn.Sequential(
            nn.Linear(self.in_channels, int(hidden_channels)),
            nn.ReLU(),
            final,
        )

    def execute(self, features: jt.Var) -> jt.Var:
        if features.shape[-1] != self.in_channels:
            raise ValueError(
                f"features last dim {features.shape[-1]} != in_channels {self.in_channels}"
            )
        return self.net(features).squeeze(-1)


def clean_bbox_normalize_np(
    cloud: np.ndarray,
    clean_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Normalize a cloud by the clean bbox center and L2-max scale."""
    cloud = _as_cloud("cloud", cloud)
    clean_ref = _as_cloud("clean_ref", clean_ref)
    p_max = clean_ref.max(axis=0)
    p_min = clean_ref.min(axis=0)
    center = ((p_max + p_min) * 0.5).astype(np.float32)
    shifted_clean = clean_ref - center
    scale = float(np.sqrt((shifted_clean * shifted_clean).sum(axis=1).max()))
    if scale < 1e-12:
        scale = 1.0
    return ((cloud - center) / scale).astype(np.float32), center, scale


def local_soft_y2x_loss_jt(
    candidates: jt.Var,
    clean: jt.Var,
    scores: jt.Var,
    k: int = 16,
    sigma: float = 0.02,
    eps: float = 1e-8,
) -> jt.Var:
    """Local-K soft clean->candidate coverage loss."""
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates must be (M,3), got {candidates.shape}")
    if clean.ndim != 2 or clean.shape[1] != 3:
        raise ValueError(f"clean must be (N,3), got {clean.shape}")
    if scores.ndim != 1 or scores.shape[0] != candidates.shape[0]:
        raise ValueError(f"scores must be (M,), got {scores.shape}")
    kk = min(int(k), int(candidates.shape[0]))
    dists, idx = safe_knn(clean.unsqueeze(0), candidates.unsqueeze(0), kk)
    dists = dists[0]
    idx = idx[0]
    probs = jt.sigmoid(scores)
    p_nb = probs[idx]
    kernel = jt.exp(-dists / (2.0 * float(sigma) * float(sigma)))
    coverage = (p_nb * kernel).sum(dim=-1)
    return -jt.log(coverage + float(eps)).mean()


def local_repulsion_loss_jt(
    candidates: jt.Var,
    scores: jt.Var,
    k: int = 8,
    eta: float = 0.01,
) -> jt.Var:
    """Local probability co-selection penalty for duplicate/cluster control."""
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError(f"candidates must be (M,3), got {candidates.shape}")
    if scores.ndim != 1 or scores.shape[0] != candidates.shape[0]:
        raise ValueError(f"scores must be (M,), got {scores.shape}")
    kk = min(int(k) + 1, int(candidates.shape[0]))
    dists, idx = safe_knn(candidates.unsqueeze(0), candidates.unsqueeze(0), kk)
    dists = dists[0]
    idx = idx[0]
    if kk > 1:
        dists = dists[:, 1:]
        idx = idx[:, 1:]
    probs = jt.sigmoid(scores)
    p_i = probs.unsqueeze(-1)
    p_j = probs[idx]
    kernel = jt.exp(-dists / (2.0 * float(eta) * float(eta)))
    return (p_i * p_j * kernel).mean()


def budget_loss_jt(scores: jt.Var, target_keep_ratio: float) -> jt.Var:
    probs = jt.sigmoid(scores)
    diff = probs.mean() - float(target_keep_ratio)
    return diff * diff


def prob_diagnostics_np(scores_np: np.ndarray, keep_n: int) -> Dict:
    scores_np = np.asarray(scores_np, dtype=np.float32).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-scores_np))
    eps = 1e-8
    entropy = -(
        probs * np.log(probs + eps)
        + (1.0 - probs) * np.log(1.0 - probs + eps)
    )
    return {
        "prob_mean": float(probs.mean()),
        "prob_std": float(probs.std()),
        "prob_min": float(probs.min()),
        "prob_max": float(probs.max()),
        "entropy_p": float(entropy.mean()),
        "sum_p": float(probs.sum()),
        "sum_p_error": float(probs.sum() - float(keep_n)),
    }
