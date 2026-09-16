#!/usr/bin/env python3
"""A 榜 mock（本地模拟评测）评测工具（精确 P2S（点到面距离），专用于 dataset/mock_test）。

从历史本地评测脚本移植，保留 clean/noisy 金标准自检口径。
关键差异对比 starter_code/evaluate.py:
    - evaluate.py 用 pc_gt (clean.npy) 的 bbox 归一化 mesh, 带坐标系偏差.
    - evaluate_mock.py 用 dataset/mock_test/.../norm.json 里**真实记录的**
      center/scale 变换 mesh, 无近似误差.

金标准自检 (必须同时满足):
    - clean.npy 作为 pred: Final ≈ 100 (P2S ≈ 100)
    - noisy.npy 作为 pred: Final ≈ 0

使用 (从 starter_code/ 运行):
    # 常规 mock 评测
    python evaluate_mock.py \
        --pred_dir ../outputs/predictions/<stage>/<run_id>/pred \
        --gt_dir ../dataset/mock_test \
        --noisy_dir ../dataset/mock_test \
        --mesh_dir ../dataset/train \
        --workers 8

    # 金标准 1: clean-as-pred (应 ~100)
    python evaluate_mock.py \
        --pred_dir ../dataset/mock_test \
        --gt_dir ../dataset/mock_test \
        --noisy_dir ../dataset/mock_test \
        --mesh_dir ../dataset/train \
        --pred_filename clean.npy \
        --workers 8

    # 金标准 2: noisy-as-pred (应 ~0)
    python evaluate_mock.py \
        --pred_dir ../dataset/mock_test \
        --gt_dir ../dataset/mock_test \
        --noisy_dir ../dataset/mock_test \
        --mesh_dir ../dataset/train \
        --pred_filename noisy.npy \
        --workers 8

test_noisy 官方测试集 (无 norm.json) 评测在 当前基础方案不走此脚本 (评测边界).
"""

import argparse
import glob
import io
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.spatial import cKDTree

try:
    import point_cloud_utils as pcu
    HAS_PCU = True
except ImportError:
    HAS_PCU = False


def load_pointcloud(path):
    return np.load(path).astype(np.float64)


def _load_obj_without_trimesh(path):
    """读取公开 MOCK 所需的基础 OBJ 顶点和面，避免评测器强依赖 trimesh。"""
    vertices = []
    faces = []
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v" and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == "f" and len(parts) >= 4:
                indices = []
                for token in parts[1:]:
                    index = int(token.split("/", 1)[0])
                    indices.append(index if index > 0 else len(vertices) + index + 1)
                for i in range(1, len(indices) - 1):
                    faces.append([indices[0] - 1, indices[i] - 1, indices[i + 1] - 1])
    if not vertices or not faces:
        raise ValueError(f"OBJ mesh has no usable vertices/faces: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def load_mesh_vf(path):
    if HAS_PCU:
        _stderr = sys.stderr
        sys.stderr = io.StringIO()
        try:
            v, f = pcu.load_mesh_vf(path)
        finally:
            sys.stderr = _stderr
        return v.astype(np.float64), f.astype(np.int32)

    try:
        import trimesh
    except ImportError:
        return _load_obj_without_trimesh(path)
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int32))


def _point_to_segment_squared_distance(points, start, end):
    direction = end - start
    denominator = float(np.dot(direction, direction))
    if denominator < 1e-15:
        delta = points - start
        return (delta * delta).sum(axis=1)
    t = ((points - start) * direction).sum(axis=1) / denominator
    t = np.clip(t, 0.0, 1.0)
    delta = points - (start + t[:, None] * direction)
    return (delta * delta).sum(axis=1)


def _point_to_triangle_squared_distance(points, a, b, c):
    """Vectorized point-to-triangle distance, used when trimesh is unavailable."""
    ab = b - a
    ac = c - a
    ap = points - a
    d1 = ap @ ab
    d2 = ap @ ac
    best = np.full(points.shape[0], np.inf, dtype=np.float64)
    remaining = np.ones(points.shape[0], dtype=bool)

    mask = remaining & (d1 <= 0.0) & (d2 <= 0.0)
    best[mask] = (ap[mask] * ap[mask]).sum(axis=1)
    remaining &= ~mask

    bp = points - b
    d3 = bp @ ab
    d4 = bp @ ac
    mask = remaining & (d3 >= 0.0) & (d4 <= d3)
    best[mask] = (bp[mask] * bp[mask]).sum(axis=1)
    remaining &= ~mask

    vc = d1 * d4 - d3 * d2
    mask = remaining & (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    denominator = d1[mask] - d3[mask]
    t = np.divide(d1[mask], denominator, out=np.zeros_like(denominator), where=denominator > 1e-15)
    delta = points[mask] - (a + t[:, None] * ab)
    best[mask] = (delta * delta).sum(axis=1)
    remaining &= ~mask

    cp = points - c
    d5 = cp @ ab
    d6 = cp @ ac
    mask = remaining & (d6 >= 0.0) & (d5 <= d6)
    best[mask] = (cp[mask] * cp[mask]).sum(axis=1)
    remaining &= ~mask

    vb = d5 * d2 - d1 * d6
    mask = remaining & (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    denominator = d2[mask] - d6[mask]
    t = np.divide(d2[mask], denominator, out=np.zeros_like(denominator), where=denominator > 1e-15)
    delta = points[mask] - (a + t[:, None] * ac)
    best[mask] = (delta * delta).sum(axis=1)
    remaining &= ~mask

    va = d3 * d6 - d5 * d4
    mask = remaining & (va <= 0.0) & ((d4 - d3) >= 0.0) & ((d5 - d6) >= 0.0)
    denominator = (d4[mask] - d3[mask]) + (d5[mask] - d6[mask])
    t = np.divide(
        d4[mask] - d3[mask],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 1e-15,
    )
    delta = points[mask] - (b + t[:, None] * (c - b))
    best[mask] = (delta * delta).sum(axis=1)
    remaining &= ~mask

    denominator = va + vb + vc
    mask = remaining & (denominator > 1e-15)
    v = vb[mask] / denominator[mask]
    w = vc[mask] / denominator[mask]
    delta = points[mask] - (a + v[:, None] * ab + w[:, None] * ac)
    best[mask] = (delta * delta).sum(axis=1)
    remaining &= ~mask

    if remaining.any():
        best[remaining] = np.minimum.reduce([
            _point_to_segment_squared_distance(points[remaining], a, b),
            _point_to_segment_squared_distance(points[remaining], a, c),
            _point_to_segment_squared_distance(points[remaining], b, c),
        ])
    return best


def load_norm(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    center = np.asarray(data["center"], dtype=np.float64)
    scale = float(data["scale"])
    if scale < 1e-12:
        raise ValueError(f"invalid norm scale in {path}: {scale}")
    return center, scale


def normalize_to_unit_sphere(pc):
    center = (pc.max(axis=0) + pc.min(axis=0)) / 2.0
    pc_centered = pc - center
    scale = np.sqrt((pc_centered ** 2).sum(axis=1)).max()
    if scale < 1e-12:
        return pc_centered, center, scale
    return pc_centered / scale, center, scale


def chamfer_distance(pc_a, pc_b, normalize=True):
    """Chamfer, 可选对 pc_b bbox 归一化后把 pc_a 带过去算距离."""
    if normalize:
        pc_b, center, scale = normalize_to_unit_sphere(pc_b)
        if scale < 1e-12:
            return 0.0
        pc_a = (pc_a - center) / scale
    tree_b = cKDTree(pc_b)
    dist_a2b, _ = tree_b.query(pc_a, k=1)
    tree_a = cKDTree(pc_a)
    dist_b2a, _ = tree_a.query(pc_b, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())


def point_to_surface_distance(pc, mesh_v, mesh_f, center, scale):
    """精确 P2S: 用 norm.json 的 center/scale 把 mesh 变到单位球空间后算 p2s.

    这是公开 MOCK 评测的核心约束: mock 点云已经是
        (orig_pc - center) / scale
    而 clean.npy 也是同样 pipeline 产出; 所以 mesh 也应用同一 (center, scale)
    变换, 二者才在同一坐标系里.
    """
    vertices = (mesh_v - center) / scale
    if HAS_PCU:
        dists, _, _ = pcu.closest_points_on_mesh(
            pc.astype(np.float32), vertices.astype(np.float32), mesh_f
        )
        return float((dists ** 2).mean())
    squared_distances = np.full(pc.shape[0], np.inf, dtype=np.float64)
    for face in mesh_f:
        face_distances = _point_to_triangle_squared_distance(
            pc, vertices[face[0]], vertices[face[1]], vertices[face[2]]
        )
        squared_distances = np.minimum(squared_distances, face_distances)
    return float(squared_distances.mean())


def metric_to_score(val_pred, val_noisy):
    if val_noisy < 1e-15:
        return 100.0 if val_pred < 1e-15 else 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - val_pred / val_noisy)))


def find_samples(base_dir, filename):
    samples = {}
    for path in sorted(glob.glob(os.path.join(base_dir, "**", filename), recursive=True)):
        rel = os.path.relpath(os.path.dirname(path), base_dir).replace("\\", "/")
        samples[rel] = path
    return samples


def find_meshes(mesh_dir, data_name="models/model_normalized.obj"):
    meshes = {}
    for path in sorted(glob.glob(os.path.join(mesh_dir, "**", data_name), recursive=True)):
        p = path
        for _ in data_name.split("/"):
            p = os.path.dirname(p)
        rel = os.path.relpath(p, mesh_dir).replace("\\", "/")
        meshes[rel] = path
    return meshes


def evaluate_single(task):
    key, pred_path, gt_path, noisy_path, mesh_path, norm_path = task

    # 输入质量硬检查: 点数/dtype/有限性, 不合格直接失败而不是静默跳过.
    try:
        pc_pred = load_pointcloud(pred_path)
        pc_gt = load_pointcloud(gt_path)
        pc_noisy = load_pointcloud(noisy_path)
    except Exception:
        # 任一文件加载失败都记 0 分, error=True.
        return (key, None, None, 0.0, None, None, 0.0, True)

    # 基础形状: 必须 (N, 3).
    if pc_pred.ndim != 2 or pc_pred.shape[-1] != 3:
        return (key, None, None, 0.0, None, None, 0.0, True)
    if pc_gt.ndim != 2 or pc_gt.shape[-1] != 3:
        return (key, None, None, 0.0, None, None, 0.0, True)
    if pc_noisy.ndim != 2 or pc_noisy.shape[-1] != 3:
        return (key, None, None, 0.0, None, None, 0.0, True)
    # finite
    if not np.isfinite(pc_pred).all():
        return (key, None, None, 0.0, None, None, 0.0, True)

    # 点数硬校验
    if pc_pred.shape != pc_noisy.shape:
        return (key, None, None, 0.0, None, None, 0.0, True)

    cd_pred = chamfer_distance(pc_pred, pc_gt, normalize=True)
    cd_noisy = chamfer_distance(pc_noisy, pc_gt, normalize=True)
    cd_score = metric_to_score(cd_pred, cd_noisy)

    mv, mf = load_mesh_vf(mesh_path)
    center, scale = load_norm(norm_path)
    p2s_pred = point_to_surface_distance(pc_pred, mv, mf, center, scale)
    p2s_noisy = point_to_surface_distance(pc_noisy, mv, mf, center, scale)
    p2s_score = metric_to_score(p2s_pred, p2s_noisy)

    return key, cd_pred, cd_noisy, cd_score, p2s_pred, p2s_noisy, p2s_score, False


def main():
    p = argparse.ArgumentParser(description="mock 评测 (精确 P2S)")
    p.add_argument("--pred_dir", required=True,
                   help="预测根目录. 金标准测试时可指向 dataset/mock_test")
    p.add_argument("--gt_dir", required=True,
                   help="通常 = dataset/mock_test (clean.npy 所在)")
    p.add_argument("--noisy_dir", required=True,
                   help="通常 = dataset/mock_test (noisy.npy 所在)")
    p.add_argument("--mesh_dir", required=True,
                   help="mesh 根目录, 通常 = dataset/train (原始 .obj 所在)")
    p.add_argument("--pred_filename", default="denoised.npy")
    p.add_argument("--gt_filename", default="clean.npy")
    p.add_argument("--noisy_filename", default="noisy.npy")
    p.add_argument("--mesh_data_name", default="models/model_normalized.obj")
    p.add_argument("--norm_filename", default="norm.json")
    p.add_argument("--workers", type=int, default=max(1, cpu_count() // 2))
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--allow-missing-mesh-norm", action="store_true",
                   help="如指定, 缺 mesh/norm 的样本记 0 分并继续; "
                        "默认 fail-loudly (mock 评测要求严格有 mesh+norm).")
    p.add_argument("--datalist", default="",
                   help="可选 datalist.txt (每行 'shapenet/<cls>/<id>'). "
                        "若指定, 只评测 datalist 里的样本集合, 不再对 gt_dir "
                        "下全量样本打 missing_pred=0 分. 用于 subset / smoke 评测.")
    args = p.parse_args()

    start = time.time()
    pred = find_samples(args.pred_dir, args.pred_filename)
    gt = find_samples(args.gt_dir, args.gt_filename)
    noisy = find_samples(args.noisy_dir, args.noisy_filename)
    meshes = find_meshes(args.mesh_dir, args.mesh_data_name)
    norms = find_samples(args.gt_dir, args.norm_filename)

    # 如果指定 datalist, 限制 keys 只包含 datalist 里的条目;
    # 否则按 gt_dir 全量样本评测.
    if args.datalist:
        if not os.path.exists(args.datalist):
            sys.exit(f"[FAIL] datalist not found: {args.datalist}")
        with open(args.datalist, "r", encoding="utf-8") as f:
            datalist_keys = set(l.strip() for l in f if l.strip())
        if not datalist_keys:
            sys.exit(f"[FAIL] datalist is empty: {args.datalist}")
        not_in_datalist = datalist_keys - (set(gt) & set(noisy))
        if not_in_datalist:
            sys.exit(f"[FAIL] {len(not_in_datalist)} datalist entries missing "
                     f"clean/noisy input (e.g. {sorted(not_in_datalist)[:3]})")
        keys = sorted(datalist_keys)
    else:
        incomplete_pairs = set(gt) ^ set(noisy)
        if incomplete_pairs:
            sys.exit(f"[FAIL] {len(incomplete_pairs)} incomplete clean/noisy pairs "
                     f"(e.g. {sorted(incomplete_pairs)[:3]})")
        keys = sorted(set(gt) & set(noisy))

    missing_pred = sorted(set(keys) - set(pred))
    common = sorted(set(keys) & set(pred))

    tasks = []
    skipped = []
    for key in common:
        mesh_path = meshes.get(key)
        norm_path = norms.get(key)
        if not mesh_path or not norm_path:
            skipped.append(key)
            continue
        tasks.append((key, pred[key], gt[key], noisy[key], mesh_path, norm_path))

    if skipped and not args.allow_missing_mesh_norm:
        msg = [
            f"[FAIL] {len(skipped)} samples missing mesh/norm (mock 评测严格要求):",
        ]
        for k in skipped[:5]:
            msg.append(f"  {k}")
        msg.append("Use --allow-missing-mesh-norm to score them as 0 instead.")
        sys.exit("\n".join(msg))

    if not tasks and not skipped:
        raise RuntimeError("no valid samples to evaluate")

    print(f"开始评测 {len(tasks)} 个样本... (P2S 后端: {'pcu-BVH' if HAS_PCU else 'numpy-triangle-fallback'})")

    workers = max(1, min(args.workers, len(tasks)))
    if workers == 1:
        results = [evaluate_single(t) for t in tasks]
    else:
        with Pool(workers) as pool:
            results = pool.map(evaluate_single, tasks)

    cd_scores, p2s_scores = [], []
    cd_preds, cd_noisys, p2s_preds, p2s_noisys = [], [], [], []
    shape_errors = []

    for r in results:
        key, cd_pred_v, cd_noisy_v, cd_s, p2s_pred_v, p2s_noisy_v, p2s_s, shape_err = r
        if shape_err:
            shape_errors.append(key)
            cd_scores.append(0.0)
            p2s_scores.append(0.0)
            continue
        cd_scores.append(cd_s)
        p2s_scores.append(p2s_s)
        cd_preds.append(cd_pred_v)
        cd_noisys.append(cd_noisy_v)
        p2s_preds.append(p2s_pred_v)
        p2s_noisys.append(p2s_noisy_v)
        if args.verbose:
            print(f"  {key}  CD_score={cd_s:.2f}  P2S_score={p2s_s:.2f}")

    for _ in missing_pred:
        cd_scores.append(0.0)
        p2s_scores.append(0.0)

    # 缺 mesh/norm 的样本 (当 --allow-missing-mesh-norm 时来到这里) 也记 0 分,
    # 不然 mean 只在成功的 tasks 里平均, 会抬高分数.
    for _ in skipped:
        cd_scores.append(0.0)
        p2s_scores.append(0.0)

    if shape_errors:
        print(f"\n[警告] {len(shape_errors)} 个样本点数不匹配，已记为 0 分：")
        for k in shape_errors:
            print(f"  {k}")

    total_samples = len(keys)
    cd_score = float(np.mean(cd_scores)) if cd_scores else 0.0
    p2s_score = float(np.mean(p2s_scores)) if p2s_scores else 0.0
    final = 0.5 * cd_score + 0.5 * p2s_score

    cd_pred_arr = np.array(cd_preds)
    cd_noisy_arr = np.array(cd_noisys)
    p2s_pred_arr = np.array(p2s_preds)
    p2s_noisy_arr = np.array(p2s_noisys)

    print("\n" + "=" * 65)
    print("  点云降噪评测结果  [evaluate_mock — 精确 P2S / 基于 norm.json]")
    print("=" * 65)
    print(f"  评测样本总数:       {total_samples}")
    print(f"  有效预测数:         {len(results)}")
    print(f"  缺失预测数:         {len(missing_pred)}")
    print(f"  点数不匹配数:       {len(shape_errors)}")
    print(f"  缺失 mesh/norm 数:  {len(skipped)}")
    print(f"  并行进程数:         {workers}")
    print(f"  评测耗时:           {time.time() - start:.1f}s")
    print("-" * 65)
    if len(cd_pred_arr):
        print(f"  平均 CD_pred:       {cd_pred_arr.mean():.8f}")
        print(f"  平均 CD_noisy:      {cd_noisy_arr.mean():.8f}")
    print(f"  CD 得分:            {cd_score:.2f} / 100.00")
    if len(p2s_pred_arr):
        print(f"  平均 P2S_pred:      {p2s_pred_arr.mean():.8f}")
        print(f"  平均 P2S_noisy:     {p2s_noisy_arr.mean():.8f}")
    print(f"  P2S 得分:           {p2s_score:.2f} / 100.00")
    print("-" * 65)
    print(f"  最终得分 (0.5×CD + 0.5×P2S):  {final:.2f} / 100.00")
    print("=" * 65)

    return final


if __name__ == "__main__":
    main()
