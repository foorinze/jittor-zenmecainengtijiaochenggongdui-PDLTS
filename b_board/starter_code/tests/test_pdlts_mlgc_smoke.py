"""PDLTS Light MLGC smoke 测试（Jittor 侧）

运行位置: WSL, 激活 jittor env
运行方式:
    cd <repo_root>/b_board/starter_code
    python -m tests.test_pdlts_mlgc_smoke

或:
    pytest starter_code/tests/test_pdlts_mlgc_smoke.py -v

验证内容（实现约定）:
    1. 四个 MLGC 类的 forward shape / dtype 正确
    2. 所有参数都能在 backward 后拿到 grad（无静默 detached）
    3. eval 模式下同一输入两次 forward 输出一致（无随机性残留）
    4. knn_group 用 jt.misc.knn 和手写 brute-force 的 idx gather 后结果一致

单测 shape 刻意小（B=2, N=64, K=8），不走 patch=1024 大规模，目的是快速定位错误。
"""

import sys
import os

# 允许从 starter_code/ 直接 `python -m tests.xxx` 运行
_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt
from jittor import nn

from src.model.pdlts_light.layer import (
    knn_group,
    get_knn_idx,
    FullyConnectedLayer,
    noiseEdgeConv,
    PreConv,
    EdgeConv,
    FeatMergeUnit,
)


def _setup():
    """固定随机种子，启用 CUDA（若可用）。"""
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


def _assert_shape(var, expected_shape, name):
    actual = tuple(var.shape)
    expected = tuple(expected_shape)
    assert actual == expected, f"{name}: expected shape {expected}, got {actual}"


def _all_params_have_grad(module: nn.Module, name: str):
    """检查 module 所有参数都拿到了非零 grad（至少 opt_grad 不为 None 且非全零）。"""
    missing = []
    zero = []
    for pname, p in module.named_parameters():
        # Jittor 的 grad 通过 module.parameters() 的 .opt_grad 或 optimizer 访问
        # 这里退而用 jt.grad 显式算
        # -- 实际检查逻辑放到调用方（单个 test 里用 jt.grad）
        pass


def test_knn_group_shape():
    """knn_group: (B, N, C) + (B, M, k) -> (B, M, k, C)."""
    B, N, C, M, k = 2, 16, 5, 8, 4
    x = jt.randn(B, N, C)
    # 构造合法 idx: 0..N-1 范围
    raw = np.random.randint(0, N, size=(B, M, k)).astype(np.int32)
    i = jt.array(raw)
    y = knn_group(x, i)
    _assert_shape(y, (B, M, k, C), "knn_group output")
    # 抽样验证数值一致：y[b, m, kk, :] == x[b, i[b, m, kk], :]
    for b in range(B):
        for m in range(M):
            for kk in range(k):
                idx_val = raw[b, m, kk]
                lhs = y[b, m, kk].numpy()
                rhs = x[b, idx_val].numpy()
                np.testing.assert_allclose(lhs, rhs, atol=1e-6,
                    err_msg=f"knn_group mismatch at b={b},m={m},k={kk}")
    print("[PASS] test_knn_group_shape")


def test_get_knn_idx_vs_brute_force():
    """get_knn_idx 和 numpy brute-force 手算 idx 经 gather 后结果一致。"""
    B, N, C, k = 2, 32, 3, 8
    x = jt.randn(B, N, C)
    idx = get_knn_idx(k=k, f=x, q=None, offset=0)
    _assert_shape(idx, (B, N, k), "get_knn_idx output shape")

    # numpy 端重算
    xnp = x.numpy()
    # dist[b, m, n] = sum((x[b,m] - x[b,n])**2)
    diff = xnp[:, :, None, :] - xnp[:, None, :, :]  # (B, M=N, N, C)
    dist = (diff ** 2).sum(-1)  # (B, N, N)
    ref_idx = np.argsort(dist, axis=-1)[..., :k]  # (B, N, k)

    # 比较 gathered feature，不比较 idx 本身（等距邻居会让 idx 顺序不同）
    jt_gather = knn_group(x, idx).numpy()
    ref_gather = np.take_along_axis(
        xnp[:, None, :, :].repeat(N, axis=1),  # (B, N, N, C)
        ref_idx[..., None].repeat(C, axis=-1),
        axis=2,
    )
    # 对每个 query 点 m 的 k 个邻居集合做排序后比较（忽略顺序）
    for b in range(B):
        for m in range(N):
            jt_set = np.sort(jt_gather[b, m].sum(-1))
            ref_set = np.sort(ref_gather[b, m].sum(-1))
            np.testing.assert_allclose(jt_set, ref_set, atol=1e-5,
                err_msg=f"brute-force vs jt.topk diverged at b={b},m={m}")
    print("[PASS] test_get_knn_idx_vs_brute_force")


def test_jt_misc_knn_vs_brute_force():
    """jt.misc.knn (主流程用) 与 brute-force 的 gathered feature 应一致。"""
    B, N, C, k = 2, 64, 3, 8
    x = jt.randn(B, N, C)
    # 主流程用: jt.misc.knn(x, x, K)
    _, jt_idx = jt.misc.knn(x, x, k)  # (B, N, k)
    _assert_shape(jt_idx, (B, N, k), "jt.misc.knn output")
    # brute-force 参考
    ref_idx = get_knn_idx(k=k, f=x, q=None, offset=0)

    # 比较两种方法 gather 后按和排序的邻居集合（idx 顺序不定，但集合要一样）
    g1 = knn_group(x, jt_idx).numpy()
    g2 = knn_group(x, ref_idx).numpy()
    for b in range(B):
        for m in range(N):
            s1 = np.sort(g1[b, m].sum(-1))
            s2 = np.sort(g2[b, m].sum(-1))
            np.testing.assert_allclose(s1, s2, atol=1e-5,
                err_msg=f"jt.misc.knn vs brute-force mismatch at b={b},m={m}")
    print("[PASS] test_jt_misc_knn_vs_brute_force")


def _forward_and_grad(module, inputs, output_reduce=lambda y: y.sum()):
    """跑一次 forward + 对 sum() backward，检查 module 参数全拿到 grad。"""
    y = module(*inputs)
    if isinstance(y, tuple):
        y = y[0]
    loss = output_reduce(y)
    # 收集所有参数
    params = list(module.parameters())
    assert len(params) > 0, "module has no trainable parameters"
    grads = jt.grad(loss, params)
    missing = []
    zero = []
    for (name, p), g in zip(module.named_parameters(), grads):
        # Jittor 会把 BatchNorm 的运行统计量列在 named_parameters() 中。
        # running_mean / running_var 不是可训练参数，没有梯度是正确行为。
        if name.endswith("running_mean") or name.endswith("running_var"):
            continue
        if g is None:
            missing.append(name)
        else:
            if float(jt.abs(g).sum().item()) < 1e-12:
                zero.append(name)
    return y, missing, zero


def test_fully_connected_layer():
    B, N, C_in, C_out = 2, 64, 16, 32
    x = jt.randn(B, N, C_in)
    for act in [None, "relu", "elu", "lrelu"]:
        m = FullyConnectedLayer(C_in, C_out, activation=act)
        y, missing, zero = _forward_and_grad(m, (x,))
        _assert_shape(y, (B, N, C_out), f"FullyConnectedLayer(act={act}) out")
        assert not missing, f"missing grad for: {missing}"
        assert not zero, f"zero grad for: {zero}"
    print("[PASS] test_fully_connected_layer")


def test_noiseEdgeConv():
    B, N, C_in, hidden, C_out, k = 2, 64, 3, 32, 48, 8
    x = jt.randn(B, N, C_in)
    _, idx = jt.misc.knn(x, x, k)
    m = noiseEdgeConv(in_channel=C_in, hidden_channel=hidden, out_channel=C_out)
    y, missing, zero = _forward_and_grad(m, (x, idx))
    _assert_shape(y, (B, N, C_out), "noiseEdgeConv out")
    assert not missing, f"missing grad for: {missing}"
    assert not zero, f"zero grad for: {zero}"
    print("[PASS] test_noiseEdgeConv")


def test_PreConv():
    B, N, C_in, C_out, k = 2, 64, 3, 16, 8
    x = jt.randn(B, N, C_in)
    _, idx = jt.misc.knn(x, x, k)
    m = PreConv(in_channel=C_in, out_channel=C_out)
    y, missing, zero = _forward_and_grad(m, (x, idx))
    _assert_shape(y, (B, N, C_out), "PreConv out")
    assert not missing, f"missing grad for: {missing}"
    assert not zero, f"zero grad for: {zero}"
    print("[PASS] test_PreConv")


def test_EdgeConv_concat_true():
    B, N, C_in, hidden, C_out, k = 2, 64, 16, 64, 32, 8
    x = jt.randn(B, N, C_in)
    _, idx = jt.misc.knn(x, x, k)
    m = EdgeConv(in_channel=C_in, hidden_channel=hidden, out_channel=C_out, concat=True)
    y, missing, zero = _forward_and_grad(m, (x, idx))
    _assert_shape(y, (B, N, C_out + C_in), "EdgeConv(concat=True) out")
    assert not missing, f"missing grad for: {missing}"
    assert not zero, f"zero grad for: {zero}"
    print("[PASS] test_EdgeConv_concat_true")


def test_EdgeConv_concat_false():
    B, N, C_in, hidden, C_out, k = 2, 64, 16, 64, 96, 8
    x = jt.randn(B, N, C_in)
    _, idx = jt.misc.knn(x, x, k)
    m = EdgeConv(in_channel=C_in, hidden_channel=hidden, out_channel=C_out, concat=False)
    y, missing, zero = _forward_and_grad(m, (x, idx))
    # concat=False 时原仓库把 hidden_channel+=32，但最终输出还是 out_channel
    _assert_shape(y, (B, N, C_out), "EdgeConv(concat=False) out")
    assert not missing, f"missing grad for: {missing}"
    assert not zero, f"zero grad for: {zero}"
    print("[PASS] test_EdgeConv_concat_false")


def test_FeatMergeUnit():
    B, N, C_in, hidden, C_out = 2, 64, 48, 64, 35  # 35 = 3 (pc_channel) + 32 (aug_channel)
    x = jt.randn(B, N, C_in)
    m = FeatMergeUnit(in_channel=C_in, hidden_channel=hidden, out_channel=C_out)
    y, missing, zero = _forward_and_grad(m, (x,))
    _assert_shape(y, (B, N, C_out), "FeatMergeUnit out")
    assert not missing, f"missing grad for: {missing}"
    assert not zero, f"zero grad for: {zero}"
    print("[PASS] test_FeatMergeUnit")


def test_eval_determinism():
    """eval 模式下同一输入两次 forward 输出应一致（不含 dropout / BN 训练模式切换带来的随机）。"""
    B, N, C_in, C_out, k = 2, 64, 3, 16, 8
    x = jt.randn(B, N, C_in)
    _, idx = jt.misc.knn(x, x, k)

    for Cls, args in [
        (PreConv, dict(in_channel=C_in, out_channel=C_out)),
        (EdgeConv, dict(in_channel=C_in, hidden_channel=32, out_channel=C_out, concat=False)),
        (FeatMergeUnit, dict(in_channel=C_in, hidden_channel=32, out_channel=C_out)),
    ]:
        m = Cls(**args)
        m.eval()
        if Cls is FeatMergeUnit:
            y1 = m(x).numpy()
            y2 = m(x).numpy()
        else:
            y1 = m(x, idx).numpy()
            y2 = m(x, idx).numpy()
        np.testing.assert_allclose(y1, y2, atol=1e-6,
            err_msg=f"eval-mode forward not deterministic for {Cls.__name__}")
    print("[PASS] test_eval_determinism")


def test_end_to_end_mlgc_pipeline():
    """按 PD-LTS DenoiseFlow.feat_extract 的次序跑一遍 MLGC。

    PreConv -> EdgeConv(concat=T) * 6 -> EdgeConv(concat=F) at i=7 -> ... (简化版 2 层)
    验证 shape 从 PreConv 输出开始能正确贯穿几层 EdgeConv + FeatMergeUnit。
    """
    B, N, k = 2, 64, 8
    xyz = jt.randn(B, N, 3)
    _, idx = jt.misc.knn(xyz, xyz, k)

    pre = PreConv(in_channel=3, out_channel=16)
    ec1 = EdgeConv(in_channel=16, hidden_channel=64, out_channel=32, concat=True)
    # 下一层 in = 16 + 32 = 48 (PD-LTS light 的 in_channelE[1] = 48)
    ec2 = EdgeConv(in_channel=48, hidden_channel=64, out_channel=32, concat=True)
    fmu = FeatMergeUnit(in_channel=80, hidden_channel=64, out_channel=3 + 32)

    f = pre(xyz, idx)
    _assert_shape(f, (B, N, 16), "pre out")
    f = ec1(f, idx)
    _assert_shape(f, (B, N, 48), "ec1 out (concat=True: 32+16)")
    f = ec2(f, idx)
    _assert_shape(f, (B, N, 80), "ec2 out (concat=True: 32+48)")
    inj = fmu(f)
    _assert_shape(inj, (B, N, 35), "fmu out (3 + 32 aug_channel)")

    # 整条链 backward
    loss = inj.sum()
    params = (list(pre.parameters()) + list(ec1.parameters())
              + list(ec2.parameters()) + list(fmu.parameters()))
    grads = jt.grad(loss, params)
    assert all(g is not None for g in grads), "some params got None grad in pipeline"
    print("[PASS] test_end_to_end_mlgc_pipeline")


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("MLGC smoke test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    test_knn_group_shape()
    test_get_knn_idx_vs_brute_force()
    test_jt_misc_knn_vs_brute_force()
    test_fully_connected_layer()
    test_noiseEdgeConv()
    test_PreConv()
    test_EdgeConv_concat_true()
    test_EdgeConv_concat_false()
    test_FeatMergeUnit()
    test_eval_determinism()
    test_end_to_end_mlgc_pipeline()
    print("=" * 60)
    print("ALL MLGC SMOKE TESTS PASSED")
    print("=" * 60)
