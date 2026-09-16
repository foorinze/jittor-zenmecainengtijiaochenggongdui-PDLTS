"""safe_knn 回归测试。

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_safe_knn

或:
    pytest starter_code/tests/test_safe_knn.py -v

背景（详见 src/model/pdlts_light/layer.py::safe_knn 文档字符串与公开实验复盘）：jittor 1.3.10 的
`jt.misc.knn` 自定义 CUDA 内核用 `auto_parallel(2, block_num=256)` 并行化，
内核对 `index`（= b*n 展平后的线程号）没有 `index >= n` 的越界守卫。当
`b * n` 不能被 256 整除时，最后一个 block 里超出范围的线程仍会执行并向
输出缓冲区之后的显存写数据，腐蚀相邻分配，最终在 predict 时随机触发
`cudaErrorIllegalAddress`（历史连续崩溃案例的根因）。

`safe_knn` 是 `jt.misc.knn` 的仓库内安全副本：内核逻辑、并行方式、数值
结果完全一致，只多一行 `if (index >= n) return;` 守卫。

本测试的目标不是复现 GPU 显存野写本身（那需要真实 CUDA 环境且发作与否
依赖进程显存布局，不适合做确定性单测），而是钉死两件更容易长期回归的
事情：
    1. 数值正确性 —— safe_knn 在多组「b*n 不能被 256 整除」的形状下，
       输出与 brute-force numpy 参考完全一致（守卫分支不改变合法线程
       的计算结果）。
    2. 与原版语义等价 —— 在 b*n 恰好整除 256（安全区，jt.misc.knn 不会
       越界）的形状下，safe_knn 与 jt.misc.knn 逐位一致，确认「仅加一行
       守卫，数值完全不变」的文档声明。
"""

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt

from src.model.pdlts_light.layer import safe_knn


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


def _brute_force_knn_np(unknown_np: np.ndarray, known_np: np.ndarray, k: int):
    """numpy 参考实现：逐 batch 算平方距离 + argsort 取前 k。

    unknown_np: (b, n, 3)
    known_np:   (b, m, 3)
    returns: (dists2, idx)，形状均为 (b, n, k)，dists2 为平方距离（不开根号，
             与官方 jt.misc.knn / safe_knn 的返回契约一致）。
    """
    b, n, _ = unknown_np.shape
    _, m, _ = known_np.shape
    dists2 = np.empty((b, n, k), dtype=np.float32)
    idx = np.empty((b, n, k), dtype=np.int32)
    for bi in range(b):
        diff = unknown_np[bi][:, None, :] - known_np[bi][None, :, :]  # (n, m, 3)
        d2 = (diff ** 2).sum(axis=-1)  # (n, m)
        order = np.argsort(d2, axis=-1, kind="stable")[:, :k]  # (n, k)
        dists2[bi] = np.take_along_axis(d2, order, axis=-1)
        idx[bi] = order
    return dists2, idx


def _compare_against_bruteforce(b: int, n: int, m: int, k: int, seed: int):
    """safe_knn vs numpy brute-force：比较按距离排序后的 (dist, idx) 对集合。

    与仓库既有 test_pdlts_mlgc_smoke.py 的判定思路一致：等距邻居可能导致 idx
    顺序不同（tie-breaking），因此逐 query 比较排序后的 (dist, idx) 元组
    多重集合，而不是逐位比较 idx 数组本身。
    """
    rng = np.random.RandomState(seed)
    unknown_np = rng.randn(b, n, 3).astype(np.float32)
    known_np = rng.randn(b, m, 3).astype(np.float32)

    unknown = jt.array(unknown_np)
    known = jt.array(known_np)
    dists2_jt, idx_jt = safe_knn(unknown, known, k)

    assert tuple(dists2_jt.shape) == (b, n, k), (
        f"dists2 shape mismatch: got {tuple(dists2_jt.shape)}, want {(b, n, k)}"
    )
    assert tuple(idx_jt.shape) == (b, n, k), (
        f"idx shape mismatch: got {tuple(idx_jt.shape)}, want {(b, n, k)}"
    )

    dists2_np_ref, idx_np_ref = _brute_force_knn_np(unknown_np, known_np, k)
    dists2_np = dists2_jt.numpy()
    idx_np = idx_jt.numpy()

    for bi in range(b):
        for ni in range(n):
            got = sorted(zip(
                np.round(dists2_np[bi, ni], 4).tolist(),
                idx_np[bi, ni].tolist(),
            ))
            ref = sorted(zip(
                np.round(dists2_np_ref[bi, ni], 4).tolist(),
                idx_np_ref[bi, ni].tolist(),
            ))
            # 只比较距离值集合（tie 时哪个 idx 命中不保证一致，但距离必须一致）
            got_d = [d for d, _ in got]
            ref_d = [d for d, _ in ref]
            np.testing.assert_allclose(
                got_d, ref_d, atol=1e-3,
                err_msg=f"b={b},n={n},m={m},k={k}: dist mismatch at bi={bi},ni={ni}"
            )
            # idx 命中的坐标必须确实对应该距离（用坐标反查校验，规避 tie 顺序问题）
            for (d, i) in got:
                coord_dist = float(((unknown_np[bi, ni] - known_np[bi, i]) ** 2).sum())
                assert abs(coord_dist - d) < 1e-2, (
                    f"b={b},n={n},m={m},k={k}: idx={i} at bi={bi},ni={ni} "
                    f"claims dist={d} but coord dist={coord_dist}"
                )

    # 也确认无 NaN/Inf，且没有越界 idx（守卫只应影响非法线程，不应产生垂悬索引）
    assert np.isfinite(dists2_np).all(), f"non-finite dists2 at b={b},n={n},m={m},k={k}"
    assert (idx_np >= 0).all() and (idx_np < m).all(), (
        f"idx out of range [0,{m}) at b={b},n={n},m={m},k={k}"
    )


# ============================================================================
# 非 256 整除形状回归（b*n 不整除 256 时旧内核会越界写入）
# ============================================================================

def test_safe_knn_non_divisible_256_small():
    """最小复现：单 batch、任意 n（几乎不可能整除 256）。"""
    _compare_against_bruteforce(b=1, n=7, m=20, k=4, seed=1)
    print("[PASS] test_safe_knn_non_divisible_256_small  b*n=7")


def test_safe_knn_non_divisible_256_medium():
    """中等规模，覆盖若干典型 b*n 值，均刻意不整除 256。"""
    cases = [
        (1, 255, 40, 5),   # b*n=255  (256-1)
        (1, 257, 40, 5),   # b*n=257  (256+1)
        (2, 129, 50, 6),   # b*n=258
        (3, 100, 60, 8),   # b*n=300
        (1, 1000, 80, 10), # b*n=1000
        (4, 63, 30, 4),    # b*n=252 (< 256, single partial block)
    ]
    for b, n, m, k in cases:
        assert (b * n) % 256 != 0, f"test setup bug: b*n={b*n} should not be divisible by 256"
        _compare_against_bruteforce(b=b, n=n, m=m, k=k, seed=b * 1000 + n)
    print(f"[PASS] test_safe_knn_non_divisible_256_medium  {len(cases)} shapes")


def test_safe_knn_denoise_crash_shape():
    """复现预测阶段的典型越界形状：b*n=586。"""
    b, n, m, k = 1, 586, 800, 32
    assert (b * n) % 256 != 0
    _compare_against_bruteforce(b=b, n=n, m=m, k=k, seed=586)
    print("[PASS] test_safe_knn_denoise_crash_shape  b*n=586 (U4 崩溃病灶形状)")


def test_safe_knn_single_point_edge_case():
    """极端边界：n=1（单点 query），b*n 恒不整除 256（除非 b 本身整除）。"""
    _compare_against_bruteforce(b=1, n=1, m=10, k=3, seed=2)
    _compare_against_bruteforce(b=5, n=1, m=10, k=3, seed=3)
    print("[PASS] test_safe_knn_single_point_edge_case")


# ============================================================================
# 256 整除（安全区）—— 确认 safe_knn 与 jt.misc.knn 逐位一致
# ============================================================================

def test_safe_knn_matches_jt_misc_knn_on_divisible_shape():
    """b*n 整除 256 时 jt.misc.knn 本身不越界，可直接与 safe_knn 逐元素比较。"""
    b, n, m, k = 1, 256, 64, 8
    assert (b * n) % 256 == 0
    rng = np.random.RandomState(7)
    unknown_np = rng.randn(b, n, 3).astype(np.float32)
    known_np = rng.randn(b, m, 3).astype(np.float32)
    unknown = jt.array(unknown_np)
    known = jt.array(known_np)

    dists2_safe, idx_safe = safe_knn(unknown, known, k)
    dists2_orig, idx_orig = jt.misc.knn(unknown, known, k)

    np.testing.assert_allclose(
        dists2_safe.numpy(), dists2_orig.numpy(), atol=1e-6,
        err_msg="safe_knn dists2 diverges from jt.misc.knn on 256-divisible shape"
    )
    np.testing.assert_array_equal(
        idx_safe.numpy(), idx_orig.numpy(),
        err_msg="safe_knn idx diverges from jt.misc.knn on 256-divisible shape"
    )
    print("[PASS] test_safe_knn_matches_jt_misc_knn_on_divisible_shape  b*n=256")


def test_safe_knn_matches_jt_misc_knn_on_multi_block_divisible_shape():
    """更大的整除形状（多个完整 256 block），确认守卫不影响任何合法 block。"""
    b, n, m, k = 4, 512, 100, 10  # b*n = 2048 = 8 * 256
    assert (b * n) % 256 == 0
    rng = np.random.RandomState(11)
    unknown_np = rng.randn(b, n, 3).astype(np.float32)
    known_np = rng.randn(b, m, 3).astype(np.float32)
    unknown = jt.array(unknown_np)
    known = jt.array(known_np)

    dists2_safe, idx_safe = safe_knn(unknown, known, k)
    dists2_orig, idx_orig = jt.misc.knn(unknown, known, k)

    np.testing.assert_allclose(
        dists2_safe.numpy(), dists2_orig.numpy(), atol=1e-6,
        err_msg="safe_knn dists2 diverges from jt.misc.knn on multi-block shape"
    )
    np.testing.assert_array_equal(
        idx_safe.numpy(), idx_orig.numpy(),
        err_msg="safe_knn idx diverges from jt.misc.knn on multi-block shape"
    )
    print("[PASS] test_safe_knn_matches_jt_misc_knn_on_multi_block_divisible_shape  b*n=2048")


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("safe_knn regression test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    test_safe_knn_non_divisible_256_small()
    test_safe_knn_non_divisible_256_medium()
    test_safe_knn_denoise_crash_shape()
    test_safe_knn_single_point_edge_case()
    test_safe_knn_matches_jt_misc_knn_on_divisible_shape()
    test_safe_knn_matches_jt_misc_knn_on_multi_block_divisible_shape()
    print("=" * 60)
    print("ALL safe_knn REGRESSION TESTS PASSED")
    print("=" * 60)
