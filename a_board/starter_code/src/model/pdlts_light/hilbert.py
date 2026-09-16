# SPDX-License-Identifier: Apache-2.0
# Adapted from PointMamba (Xiaoyang Wu, Kaixin Xu) and numpy-hilbert-curve.
# Copyright (c) 2020 Princeton Laboratory for Intelligent Probabilistic Systems
# The earlier NumPy implementation retains its MIT notice in LICENSES/.
"""Hilbert 空间填充曲线编码（numpy 实现，Skilling 2004 算法）。

来源: PointMamba 的 Hilbert 曲线实现，numpy 移植。
用途: global-context module 的 global context block 中，对 patch 内点按 Hilbert 顺序排序，
     使空间相邻的点在序列中位置也相近（locality-preserving）。

不依赖 PyTorch / Jittor，只在 numpy 上运行。排序结果作为整数 index 传入 Jittor graph
做 gather/scatter。

用法:
    import numpy as np
    from .hilbert import hilbert_encode, hilbert_sort_indices

    coords_np: np.ndarray  shape (B, N, 3)  float32, 已中心化的 patch 坐标
    bits: int = 10      每维编码比特数

    sort_idx, unsort_idx = hilbert_sort_indices(coords_np, bits=10)
    # sort_idx:   (B, N) int64, 每个 batch 内按 Hilbert 顺序排列的 index
    # unsort_idx: (B, N) int64, 恢复原始顺序的 index (= argsort(sort_idx))
"""

import numpy as np


def _right_shift(binary, k=1, axis=-1):
    """二进制数组右移 k 位，左侧补 0。"""
    if binary.shape[axis] <= k:
        return np.zeros_like(binary)
    slicing = [slice(None)] * binary.ndim
    slicing[axis] = slice(None, -k)
    shifted = binary[tuple(slicing)]
    pad_width = [(0, 0)] * binary.ndim
    pad_width[axis] = (k, 0)
    return np.pad(shifted, pad_width, mode="constant", constant_values=0)


def _binary2gray(binary, axis=-1):
    """二进制 → Gray code: X ^ (X >> 1)。"""
    shifted = _right_shift(binary, axis=axis)
    return np.logical_xor(binary, shifted)


def _gray2binary(gray, axis=-1):
    """Gray code → 二进制。"""
    bits = gray.shape[axis]
    shift = 2 ** int(np.ceil(np.log2(bits)) - 1)
    while shift > 0:
        gray = np.logical_xor(gray, _right_shift(gray, int(shift), axis=axis))
        shift = shift // 2
    return gray


def hilbert_encode(locs, num_dims=3, num_bits=10):
    """将超立方体中的坐标编码为 Hilbert 整数。

    Skilling, J. (2004). Programming the Hilbert curve.
    AIP Conference Proceedings, 707(1), 381-387.

    Args:
        locs: np.ndarray, shape (..., num_dims), dtype int, 每维范围 [0, 2**num_bits-1]
        num_dims: 维度数 (默认 3)
        num_bits: 每维编码比特数 (默认 10)

    Returns:
        hilbert_codes: np.ndarray, shape (...), dtype int64
    """
    if locs.shape[-1] != num_dims:
        raise ValueError(
            f"locs last dim must be {num_dims}, got {locs.shape[-1]}"
        )
    if num_dims * num_bits > 63:
        raise ValueError(
            f"num_dims*num_bits={num_dims * num_bits} > 63, can't encode into int64"
        )

    orig_shape = locs.shape[:-1]

    # locs: (..., num_dims) int → flatten to (N_total, num_dims)
    locs_flat = locs.reshape(-1, num_dims).astype(np.int64)
    n_total = locs_flat.shape[0]

    # 提取每维的 num_bits 个低位 bit → (N_total, num_dims, num_bits)
    gray = np.zeros((n_total, num_dims, num_bits), dtype=np.uint8)
    for bit in range(num_bits):
        gray[:, :, bit] = (locs_flat >> bit) & 1

    # 逆序 bits (MSB first)
    gray = gray[:, :, ::-1]

    # Skilling 正向迭代
    for bit in range(num_bits):
        for dim in range(num_dims):
            mask = gray[:, dim, bit].astype(bool)
            # 此 bit 为 1 → 将第 0 维的低位 bit 取反
            gray[:, 0, bit + 1:] = np.logical_xor(
                gray[:, 0, bit + 1:],
                mask[:, None]
            )
            # 此 bit 为 0 → 交换第 0 维与当前维的低位 bit
            to_flip = (
                ~mask[:, None]
                & np.logical_xor(gray[:, 0, bit + 1:], gray[:, dim, bit + 1:])
            )
            gray[:, dim, bit + 1:] = np.logical_xor(gray[:, dim, bit + 1:], to_flip)
            gray[:, 0, bit + 1:] = np.logical_xor(gray[:, 0, bit + 1:], to_flip)

    # 拉平 interleaved bits: (N_total, num_dims, num_bits) → (N_total, num_bits * num_dims)
    # 按 num_dims interleave: dim 0 bit 0, dim 1 bit 0, dim 2 bit 0, dim 0 bit 1, ...
    gray_flat = np.zeros((n_total, num_bits * num_dims), dtype=np.uint8)
    for bit in range(num_bits):
        for dim in range(num_dims):
            gray_flat[:, bit * num_dims + dim] = gray[:, dim, bit]

    hh_bin = _gray2binary(gray_flat)

    # 二进制 → int64 (big-endian)
    codes = np.zeros(n_total, dtype=np.int64)
    for b in range(num_bits * num_dims):
        codes |= hh_bin[:, b].astype(np.int64) << (num_bits * num_dims - 1 - b)

    return codes.reshape(orig_shape)


def hilbert_sort_indices(coords, bits=10, num_dims=3):
    """给定浮点坐标，返回按 Hilbert 顺序排序/恢复的 index。

    Args:
        coords: np.ndarray (B, N, 3) float32/float64, patch 坐标 (已中心化)
        bits: 每维编码位数 (默认 10, 对应 1024 格点)
        num_dims: 空间维度 (默认 3)

    Returns:
        sort_idx:   (B, N) int64, coords[b, sort_idx[b]] = Hilbert-ordered coords
        unsort_idx: (B, N) int64, 恢复原始顺序的逆 index

    坐标标准化: 每个 batch 内独立做 min-max 归一化到 [0, 2^bits-1]。
    """
    B, N, D = coords.shape
    assert D == num_dims, f"coords last dim must be {num_dims}, got {D}"

    sort_idx_list = []
    unsort_idx_list = []

    for b in range(B):
        c = coords[b]  # (N, D)
        c_min = c.min(axis=0, keepdims=True)
        c_max = c.max(axis=0, keepdims=True)
        c_range = c_max - c_min
        c_range = np.where(c_range < 1e-10, 1.0, c_range)
        c_norm = (c - c_min) / c_range
        c_int = np.floor(c_norm * ((1 << bits) - 1)).astype(np.int64)
        c_int = np.clip(c_int, 0, (1 << bits) - 1)

        codes = hilbert_encode(c_int, num_dims=num_dims, num_bits=bits)
        sort_idx = np.argsort(codes).astype(np.int64)
        unsort_idx = np.argsort(sort_idx).astype(np.int64)
        sort_idx_list.append(sort_idx)
        unsort_idx_list.append(unsort_idx)

    return np.stack(sort_idx_list, axis=0), np.stack(unsort_idx_list, axis=0)
