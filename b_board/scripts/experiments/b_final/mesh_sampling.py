"""B 榜共用 mesh（网格）工具：OBJ 读取、面积加权采样、归一化、噪声尺度估计。

本模块只提供确定性工具函数，不做任何实验判定，不写 outputs。

背景：B 榜 `train_b` 的 `model_normalized.obj` 实际未归一化（bbox 对角线中位数
约 3.79，范围 [0.356, 1693]），而 `test_noisy_b` 的点云是归一化的（bbox 中心
归一化后 max radius 约 1.03）。因此构造 mock 时必须显式归一化，并剔除被离群
顶点主导的退化网格。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover
    cKDTree = None


@dataclass(frozen=True)
class MeshHealth:
    """单个网格的体检读数。全部为归一化前的原始尺度统计。"""

    num_verts: int
    num_faces: int
    surface_area: float
    bbox_diag: float
    max_radius: float
    radius_p995: float
    outlier_ratio: float
    finite: bool

    @property
    def is_degenerate(self) -> bool:
        """面积或面数为 0，或存在非有限坐标。"""
        return (not self.finite) or self.num_faces == 0 or self.surface_area <= 0.0

    @property
    def is_outlier_dominated(self) -> bool:
        """归一化尺度被极少数离群顶点主导，归一化后主体会塌缩。"""
        return self.outlier_ratio < 0.5


def load_obj_mesh(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取 OBJ 的顶点和三角面，忽略 vt/vn；多边形面按扇形三角化。

    OBJ 索引从 1 开始，且允许负索引（相对当前顶点数倒数）。
    """
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, "r", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                idx = []
                for token in tokens:
                    raw = int(token.split("/")[0])
                    idx.append(raw - 1 if raw > 0 else len(verts) + raw)
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])
    vertices = np.asarray(verts, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    return vertices, triangles


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """每个三角面的面积。"""
    if len(faces) == 0:
        return np.zeros(0, dtype=np.float64)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cross, axis=1)


def inspect_mesh(vertices: np.ndarray, faces: np.ndarray) -> MeshHealth:
    """计算网格体检读数。

    `outlier_ratio` 定义为 顶点半径的 99.5 分位 / 最大半径（相对 bbox 中心）。
    该值接近 1 表示尺度由主体决定；显著小于 1 表示极少数离群顶点撑大了包围盒，
    按 max radius 归一化会让主体塌缩。
    """
    if len(vertices) == 0:
        return MeshHealth(0, len(faces), 0.0, 0.0, 0.0, 0.0, 0.0, False)
    finite = bool(np.isfinite(vertices).all())
    if not finite:
        vertices = vertices[np.isfinite(vertices).all(axis=1)]
        if len(vertices) == 0:
            return MeshHealth(0, len(faces), 0.0, 0.0, 0.0, 0.0, 0.0, False)
    bbox_center = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    radii = np.linalg.norm(vertices - bbox_center, axis=1)
    max_radius = float(radii.max())
    radius_p995 = float(np.percentile(radii, 99.5))
    return MeshHealth(
        num_verts=len(vertices),
        num_faces=len(faces),
        surface_area=float(triangle_areas(vertices, faces).sum()),
        bbox_diag=float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))),
        max_radius=max_radius,
        radius_p995=radius_p995,
        outlier_ratio=float(radius_p995 / max_radius) if max_radius > 0 else 0.0,
        finite=finite,
    )
def sample_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """面积加权均匀采样网格表面，返回 (num_points, 3)。

    面积加权保证采样密度与三角形大小无关，这是和 A 榜 mock 构造一致的口径。
    """
    areas = triangle_areas(vertices, faces)
    total = areas.sum()
    if total <= 0:
        raise ValueError("mesh surface area is zero, cannot sample")
    probs = areas / total
    face_idx = rng.choice(len(faces), size=num_points, p=probs)
    tri = vertices[faces[face_idx]]
    # 单位三角形上的均匀重心坐标采样
    u = rng.random((num_points, 1))
    v = rng.random((num_points, 1))
    flip = (u + v) > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    return tri[:, 0] + u * (tri[:, 1] - tri[:, 0]) + v * (tri[:, 2] - tri[:, 0])


def normalize_bbox_unit_sphere(
    points: np.ndarray, reference: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """按 bbox 中心平移、按最大半径缩放，与 A 榜 mock 的 `mesh_bbox_unit_sphere` 一致。

    `reference` 用于让 clean/noisy 或 dense target 共享同一 center/scale。
    返回 (归一化点云, center, scale)。
    """
    ref = points if reference is None else reference
    center = (ref.max(axis=0) + ref.min(axis=0)) / 2.0
    shifted = ref - center
    scale = float(np.sqrt((shifted**2).sum(axis=1).max()))
    if scale <= 0:
        raise ValueError("normalization scale is zero")
    return (points - center) / scale, center, scale


def estimate_offplane_sigma(
    points: np.ndarray,
    k: int = 24,
    num_queries: int = 4000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """用局部 PCA 最小特征值估计离面噪声厚度，返回 (中位离面 sigma, 中位最近邻间距)。

    注意：当噪声尺度接近采样间距时，该估计器无法区分"噪声厚度"和"采样稀疏"，
    绝对值不可信。必须配合 B 榜训练数据噪声标定结果
    反演使用，不要直接把估计值当作真实 sigma。
    """
    if cKDTree is None:
        raise ImportError("scipy is required for estimate_offplane_sigma")
    rng = np.random.default_rng(0) if rng is None else rng
    tree = cKDTree(points)
    num_queries = min(num_queries, len(points))
    query_idx = rng.choice(len(points), size=num_queries, replace=False)
    dists, neighbor_idx = tree.query(points[query_idx], k=k)
    patches = points[neighbor_idx]
    centered = patches - patches.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k
    eigenvalues = np.linalg.eigvalsh(cov)
    offplane = np.sqrt(np.maximum(eigenvalues[:, 0], 0.0))
    return float(np.median(offplane)), float(np.median(dists[:, 1]))


def add_laplace_noise(
    points: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """加各分量独立 Laplace 噪声，与 `AugmentAddNoise` 的分布一致。

    这里的 `sigma` 是 Laplace scale 参数 b，逐分量标准差为 sqrt(2) * b。
    """
    return points + rng.laplace(0.0, sigma, size=points.shape)
