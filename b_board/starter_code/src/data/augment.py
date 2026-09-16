from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .asset import Asset
from .spec import ConfigSpec
from .utils import random_euler_rotation, sample_vertex_groups

@dataclass(frozen=True)
class Augment(ConfigSpec):
    
    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass
    
    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):

    num_samples: int # total number of vertices on the face to be sampled

    num_vertex_samples: int=0 # number of vertices to be chosen

    # dense-surface auxiliary dense surface auxiliary：额外采样一份更稠密的 clean 表面点云，
    # 写入 asset.sampled_vertices_clean_dense。默认 0 = 不采样，零影响旧线。
    # 注意：此处采的是原始 mesh 坐标系下的 raw dense 点；归一化由后续
    # AugmentNormalizePC 用主 clean 的 center/scale 同步完成（见该类注释）。
    dense_clean_target_samples: int=0

    dense_clean_num_vertex_samples: int=0

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        assert asset.vertices is not None
        assert asset.faces is not None
        sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
            vertices=asset.vertices,
            faces=asset.faces,
            num_samples=self.num_samples,
            num_vertex_samples=self.num_vertex_samples,
        )
        asset.sampled_vertices = sampled_vertices

        # dense-surface auxiliary: 额外稠密 clean 采样（独立随机采样，与主 clean 不共享索引）。
        if self.dense_clean_target_samples > 0:
            dense_vertices, _dn, _dvg, _dh = sample_vertex_groups(
                vertices=asset.vertices,
                faces=asset.faces,
                num_samples=self.dense_clean_target_samples,
                num_vertex_samples=self.dense_clean_num_vertex_samples,
            )
            asset.sampled_vertices_clean_dense = dense_vertices

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices = pc / scale

        # dense-surface auxiliary: dense clean 必须用主 clean 的 center/scale 同步归一化，
        # 否则 dense target 与 noisy patch 不在同一坐标系（实验会失真）。
        if asset.sampled_vertices_clean_dense is not None:
            asset.sampled_vertices_clean_dense = (
                asset.sampled_vertices_clean_dense - center
            ) / scale

@dataclass(frozen=True)
class AugmentAddNoise(Augment):
    
    noise_std_min: float
    
    noise_std_max: float
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        return AugmentAddNoise(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        noise_std = np.random.uniform(self.noise_std_min, self.noise_std_max)
        noise = np.random.laplace(0, noise_std, size=pc.shape)
        asset.sampled_vertices_noisy = pc + noise

@dataclass(frozen=True)
class AugmentLinear(Augment):
    
    scale: Tuple[float, float]=(1.0, 1.0)
    
    rotate_x_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_y_range: Tuple[float, float]=(0.0, 0.0)
    
    rotate_z_range: Tuple[float, float]=(0.0, 0.0)
    
    scale_p: float=0.0
    
    rotate_p: float=0.0
    
    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)
    
    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)

@dataclass(frozen=True)
class AugmentLinearPC(Augment):
    """B 榜实验新增：作用于采样点云的旋转/缩放增广（注册名 linear_pc）。

    背景（2026-08-12 核实）：AugmentLinear 经 asset.transform() 只变换
    asset.vertices（mesh 顶点）；在 transform 链中它排在 sample 之后，
    对 sampled_vertices / sampled_vertices_noisy 是静默 no-op —— 即 A 榜
    配置中的 linear 增广从未真正生效。本类是点云版实现，新注册名，
    不改 AugmentLinear 语义，存量配置零影响。

    推荐链位（B 榜 npy_pair 动态增广）：
      npy_pair loader → linear_pc（旋转 clean 与旧 noisy）
      → add_noise（在旋转后坐标系加各向坐标轴 Laplace，与官方噪声形态一致，
                   并覆盖旧 noisy）→ patch
    """

    scale: Tuple[float, float]=(1.0, 1.0)

    rotate_x_range: Tuple[float, float]=(0.0, 0.0)

    rotate_y_range: Tuple[float, float]=(0.0, 0.0)

    rotate_z_range: Tuple[float, float]=(0.0, 0.0)

    scale_p: float=0.0

    rotate_p: float=0.0

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinearPC':
        cls.check_keys(kwargs)
        return AugmentLinearPC(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        trans = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans = r @ trans
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans = scale @ trans

        def _apply_pc(v: np.ndarray) -> np.ndarray:
            return np.matmul(v, trans[:3, :3].transpose()) + trans[:3, 3]

        if asset.sampled_vertices is not None:
            asset.sampled_vertices = _apply_pc(asset.sampled_vertices)
        if asset.sampled_vertices_noisy is not None:
            asset.sampled_vertices_noisy = _apply_pc(asset.sampled_vertices_noisy)
        if asset.sampled_vertices_clean_dense is not None:
            asset.sampled_vertices_clean_dense = _apply_pc(asset.sampled_vertices_clean_dense)

@dataclass(frozen=True)
class AugmentPatch(Augment):

    patch_size: int

    num_patches: int

    train_cvm_network: bool

    use_noisy_seed_center: bool = False

    # target construction: target 构造模式
    #   "paired_idx"                  — 旧行为，clean patch = pc[nn_idx]（默认）
    #   "clean_knn_seed"              — T1a: clean KNN around pc[seed_idx]
    #   "clean_knn_noisy_seed_nn"     — clean KNN around nn_clean(noisy_seed_coord)
    #                                   推荐方案，centroid drift 最低
    target_mode: str = "paired_idx"

    # dense-surface auxiliary dense surface auxiliary：在 dense clean 点云上取局部 target，
    # 写入 asset.meta['pc_clean_dense_local']，作为弱辅助 surface 监督。
    #   dense_target_size > 0 且 asset 有 sampled_vertices_clean_dense 时才构造。
    #   不替换 pc_clean / pc_clean_target，主 paired 路径完全不变（默认 0 零影响）。
    #   局部区域锚点 = 与主 patch 同一 noisy seed 在 dense clean 上的最近点，
    #   保证 dense local patch 和主 noisy patch 覆盖同一局部表面。
    dense_target_size: int = 0

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy

        assert pc is not None
        assert pc_noisy is not None

        _valid_modes = ("paired_idx", "clean_knn_seed", "clean_knn_noisy_seed_nn")
        if self.target_mode not in _valid_modes:
            raise ValueError(
                f"target_mode must be one of {_valid_modes}, "
                f"got {self.target_mode!r}"
            )

        _is_clean_target = self.target_mode in ("clean_knn_seed", "clean_knn_noisy_seed_nn")

        N = pc_noisy.shape[0]

        seed_idx = np.random.permutation(N)[:self.num_patches]   # (P,)
        seed_points = pc_noisy[seed_idx]                         # (P, 3)

        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)   # (P, M)

        pat_A = pc_noisy[nn_idx]  # (P, M, 3)
        pat_B = pc[nn_idx]        # (P, M, 3)

        # --- 2.0a: clean target modes ---
        if _is_clean_target:
            tree_clean = cKDTree(pc)
            if self.target_mode == "clean_knn_seed":
                # T1a: KNN around clean point at same global index as noisy seed
                clean_query_points = pc[seed_idx]
            else:  # clean_knn_noisy_seed_nn
                # nearest clean point to noisy seed coord, then KNN
                _, nn_seed_to_clean = tree_clean.query(seed_points, k=1)    # (P, 1)
                clean_query_points = pc[nn_seed_to_clean.reshape(-1)]        # (P, 3)
            _, clean_nn_idx = tree_clean.query(clean_query_points, k=self.patch_size)  # (P, M)
            pat_clean_target_raw = pc[clean_nn_idx]                                     # (P, M, 3)
        # ------------------------------------

        # --- dense-surface auxiliary: dense surface auxiliary local target ---
        # 在更稠密的 clean 点云上，以 "noisy seed 的最近 dense clean 点" 为局部锚，
        # KNN 取 dense_target_size 个点。与主 paired 路径独立，不改 pc_clean。
        _dense_aux = (
            self.dense_target_size > 0
            and asset.sampled_vertices_clean_dense is not None
        )
        if _dense_aux:
            pc_dense = asset.sampled_vertices_clean_dense          # (D, 3)
            tree_dense = cKDTree(pc_dense)
            _, dense_seed_nn = tree_dense.query(seed_points, k=1)  # (P, 1)
            dense_query_points = pc_dense[dense_seed_nn.reshape(-1)]  # (P, 3)
            _, dense_nn_idx = tree_dense.query(
                dense_query_points, k=self.dense_target_size
            )                                                       # (P, Md)
            pat_dense_target_raw = pc_dense[dense_nn_idx]           # (P, Md, 3)
        # ------------------------------------

        if self.use_noisy_seed_center:
            seed_center = seed_points[:, None, :]  # (P, 1, 3)
            pat_A = pat_A - seed_center
            pat_B = pat_B - seed_center
            pat_t = pat_A
            if _is_clean_target:
                pat_clean_target_raw = pat_clean_target_raw - seed_center
            if _dense_aux:
                pat_dense_target_raw = pat_dense_target_raw - seed_center
        else:
            l1, l2_const = 1e-8, 1.0
            t = np.random.rand(self.num_patches, self.patch_size, 1)
            t = (l2_const - l1) * t + l1

            pat_t = t * pat_B + (1 - t) * pat_A
            seed_points_t = (
                t[:, 0:1, :] * pc[seed_idx][:, None, :] +
                (1 - t[:, 0:1, :]) * pc_noisy[seed_idx][:, None, :]
            )

            pat_A = pat_A - seed_points_t
            pat_B = pat_B - seed_points_t
            pat_t = pat_t - seed_points_t
            if _is_clean_target:
                pat_clean_target_raw = pat_clean_target_raw - seed_points_t
            if _dense_aux:
                # 与主 patch 同一 seed_points_t 中心化口径，保证 dense local
                # target 与 pc_noisy 在同一局部坐标系。
                pat_dense_target_raw = pat_dense_target_raw - seed_points_t

        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_noisy'] = pat_A
        asset.meta['pc_clean'] = pat_B
        asset.meta['pc_mix'] = pat_t

        if _is_clean_target:
            asset.meta['pc_clean_target'] = pat_clean_target_raw
            asset.meta['_audit_seed_idx'] = seed_idx.astype(np.int64)
            asset.meta['_audit_noisy_nn_idx'] = nn_idx.astype(np.int64)
            asset.meta['_audit_clean_target_idx'] = clean_nn_idx.astype(np.int64)

        if _dense_aux:
            asset.meta['pc_clean_dense_local'] = pat_dense_target_raw
            asset.meta['_audit_dense_nn_idx'] = dense_nn_idx.astype(np.int64)

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "linear_pc": AugmentLinearPC,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments
