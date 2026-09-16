"""PDLTS Light 整网 smoke 单测。

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_pdlts_light_model_smoke

验证内容:
    1. forward shape 正确: (B, N, 3) -> (B, N, 3)
    2. 所有可训练参数都能拿到非零 grad (排除 BN running_stats / ActNorm is_inited / AffineCoupling mask / channel_mask)
    3. FBM non-identity: 强制把 net 的关键层初始化到非 trivial 状态后,
       denoised(noisy) ≠ noisy，证明 FBM 置零后通过 inverse 能产生真实位移
    4. Determinism: eval 模式下同一输入两次 forward 结果一致
    5. Param count 在合理范围 (log 一下总参数数, 不做断言)
    6. Shape 沿 MLGC 注入路径逐层正确: inj_f[i] 都是 (B, N, pc+aug=51)

单测刻意小 shape (B=2, N=32, num_neighbors=8), 便于快速定位错误。
本文件不测量 patch_size=1024 时的训练性能。
"""

import sys
import os
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt
from jittor import nn

from src.model.pdlts_light.model import (
    PDLTSLightNetwork,
    LIGHT_AUG_CHANNEL,
    LIGHT_CUT_CHANNEL,
    LIGHT_N_INJECTOR,
    LIGHT_NFLOW_MODULE,
)
from src.model.pdlts_light.inn import AffineCoupling


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


# 全网共用一个"小号"网络参数以控制单测时长
# num_neighbors 必须 < N
SMALL_NET_KWARGS = dict(
    num_neighbors=8,   # 测试用小 k, patch 也小
    coupling_hidden=32,
    mlgc_hidden=32,
)


def test_forward_shape_and_dtype():
    """patch (B, N, 3) 经过 forward 后 shape 不变, dtype 是 float32。"""
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    x = jt.randn(B, N, 3)
    denoised, ldj, loss_d = net(x)
    assert tuple(denoised.shape) == (B, N, 3), f"denoised shape {denoised.shape}"
    assert denoised.dtype == "float32", f"denoised dtype {denoised.dtype}"
    assert tuple(ldj.shape) == (B,), f"ldj shape {ldj.shape}"
    assert tuple(loss_d.shape) == (1,), f"loss_d shape {loss_d.shape}"
    print(f"[PASS] test_forward_shape_and_dtype  out={tuple(denoised.shape)}")


def test_injection_feature_shapes():
    """feat_extract 返回 12 个 (B, N, 51) 的 inj feature。"""
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    x = jt.randn(B, N, 3)
    _, knn_idx = jt.misc.knn(x, x, net.num_neighbors)
    inj_f = net.feat_extract(x, knn_idx)
    assert len(inj_f) == LIGHT_N_INJECTOR, f"expected {LIGHT_N_INJECTOR} inj features, got {len(inj_f)}"
    expected_inj_c = 3 + LIGHT_AUG_CHANNEL
    for i, f in enumerate(inj_f):
        assert tuple(f.shape) == (B, N, expected_inj_c), \
            f"inj_f[{i}] shape {f.shape} != (B, N, {expected_inj_c})"
    print(f"[PASS] test_injection_feature_shapes  {len(inj_f)} x (B,N,{expected_inj_c})")


def test_unit_coupling_shape():
    """unit_coupling(xyz, knn_idx) -> (B, N, aug_channel)。"""
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    x = jt.randn(B, N, 3)
    _, knn_idx = jt.misc.knn(x, x, net.num_neighbors)
    aug = net.unit_coupling(x, knn_idx)
    assert tuple(aug.shape) == (B, N, LIGHT_AUG_CHANNEL), f"aug shape {aug.shape}"
    print(f"[PASS] test_unit_coupling_shape  out=(B,N,{LIGHT_AUG_CHANNEL})")


def test_grad_flow():
    """backward 后可训练参数都应拿到非零 grad。排除状态变量、mask、running_stats。"""
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)

    # 给 AffineCoupling 的最后一层 Linear 注入小扰动,
    # 避免 net 一开始就处于 "所有 FlowAssembly 接近 identity" 状态,
    # 这时有些权重的 grad 可能非零但数值极小, 不好判 true-zero 还是 near-zero.
    np.random.seed(31)
    for mod in net.flow_assemblies:
        for layer in mod.chain:
            if isinstance(layer, AffineCoupling):
                last = layer.net[-1]
                # 维度: (out=2C, in=hidden)
                last.weight = jt.array(
                    np.random.randn(*last.weight.shape).astype(np.float32) * 0.05
                )
                last.bias = jt.array(
                    np.random.randn(*last.bias.shape).astype(np.float32) * 0.05
                )

    x = jt.randn(B, N, 3)
    # 先跑一次 forward 让 ActNorm1d 完成 first-forward init
    _ = net(x)

    # 再跑一次, 对 denoised.sum() 求 grad
    x = jt.randn(B, N, 3)
    denoised, _, _ = net(x)
    loss = denoised.sum()

    trainable_named_params = []
    for name, p in net.named_parameters():
        # 排除: BN running_stats, ActNorm1d.is_inited, AffineCoupling.mask, channel_mask
        # 这些变量会进入 state_dict，但不是优化器应更新的参数。
        if name.endswith("running_mean") or name.endswith("running_var"):
            continue
        if name.endswith("is_inited") or name.endswith("mask"):
            continue
        if name == "channel_mask":
            continue
        trainable_named_params.append((name, p))

    params = [p for _, p in trainable_named_params]
    grads = jt.grad(loss, params)
    missing = []
    zero = []
    for (name, _), g in zip(trainable_named_params, grads):
        if g is None:
            missing.append(name)
        elif float(jt.abs(g).sum().item()) < 1e-12:
            zero.append(name)
    assert not missing, f"missing grad for {len(missing)} params: {missing[:5]}..."
    assert not zero, f"zero grad for {len(zero)} params: {zero[:5]}..."
    print(f"[PASS] test_grad_flow  trainable_params_with_nonzero_grad={len(params) - len(missing) - len(zero)}")


def test_fbm_produces_nonidentity():
    """FBM: 置零最后 cut_channel 通道后, denoised ≠ noisy。

    不是严格证明"能去噪", 只是证明整网不是 pass-through。
    """
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)

    # 注入非零扰动, 避免 AffineCoupling 全 identity
    np.random.seed(77)
    for mod in net.flow_assemblies:
        for layer in mod.chain:
            if isinstance(layer, AffineCoupling):
                last = layer.net[-1]
                last.weight = jt.array(
                    np.random.randn(*last.weight.shape).astype(np.float32) * 0.1
                )
                last.bias = jt.array(
                    np.random.randn(*last.bias.shape).astype(np.float32) * 0.1
                )

    x = jt.randn(B, N, 3)
    # 第一次 forward 触发 ActNorm 初始化
    _ = net(x)

    # 用新输入再跑一次
    x = jt.randn(B, N, 3)
    denoised, _, _ = net(x)
    diff = float(jt.abs(denoised - x).max().item())
    # 要求 denoised 和 noisy 有明显差异 (> 0.01 是 "非 trivial 位移" 的宽松下界)
    assert diff > 0.01, \
        f"FBM didn't produce non-identity output: max|denoised - noisy|={diff:.4e}"
    print(f"[PASS] test_fbm_produces_nonidentity  max_diff={diff:.4e}")


def test_eval_determinism():
    """eval 模式下同一输入两次 forward 输出一致。"""
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    x = jt.randn(B, N, 3)
    # 先走一次触发 ActNorm 初始化 (否则第一次 forward 会写 weight/bias, 第二次读的值会变)
    _ = net(x)
    net.eval()
    y1, _, _ = net(x)
    y2, _, _ = net(x)
    diff = float(jt.abs(y1 - y2).max().item())
    assert diff < 1e-5, f"eval-mode determinism violated: max_diff={diff:.4e}"
    print(f"[PASS] test_eval_determinism  max_diff={diff:.4e}")


def test_fbm_channel_mask_values():
    """channel_mask 末尾 cut_channel 位必须是 0, 其余是 1。"""
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    mask_np = net.channel_mask.numpy()
    # shape: (1, 1, pc+aug)
    expected_shape = (1, 1, 3 + LIGHT_AUG_CHANNEL)
    assert mask_np.shape == expected_shape, f"mask shape {mask_np.shape}"
    # 前 (total - cut) 必须全 1
    non_cut = mask_np[..., : -LIGHT_CUT_CHANNEL]
    cut = mask_np[..., -LIGHT_CUT_CHANNEL:]
    assert (non_cut == 1.0).all(), "non-cut channels should all be 1"
    assert (cut == 0.0).all(), "cut channels should all be 0"
    print(f"[PASS] test_fbm_channel_mask_values  non_cut=all-1, cut=all-0")


def test_injected_flow_roundtrip():
    """不经过 FBM 置零时, g(f(x + aug, inj_f), inj_f) 应该能重构原始 latent。

    这个测试专门防止 f/g 的注入顺序写反: shape 即使全对, 这里也会爆。
    """
    B, N = 2, 32
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    x = jt.randn(B, N, 3)
    _, knn_idx = jt.misc.knn(x, x, net.num_neighbors)
    inj_f = net.feat_extract(x, knn_idx)
    aug = net.unit_coupling(x, knn_idx)
    x_aug = jt.concat([x, aug], dim=-1)

    z, _ = net.f(x_aug, inj_f)
    rec = net.g(z, inj_f)
    err = float(jt.abs(rec - x_aug).max().item())
    assert err < 1e-3, f"f/g roundtrip too large: max_err={err:.4e}"
    print(f"[PASS] test_injected_flow_roundtrip  max_err={err:.4e}")


def test_state_save_load_fixed_buffers():
    """固定状态必须进入 checkpoint: FBM mask 和 ActNorm 初始化状态不能丢。"""
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)
    x = jt.randn(1, 32, 3)
    _ = net(x)  # 触发 ActNorm 初始化

    state = net.state_dict()
    assert "channel_mask" in state, "channel_mask missing from state_dict"
    assert sum(k.endswith("is_inited") for k in state) == LIGHT_NFLOW_MODULE * 2, \
        "ActNorm init flags should be saved"

    fd, path = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        net.save(path)
        loaded = PDLTSLightNetwork(**SMALL_NET_KWARGS)
        loaded.load(path)
        mask_sum = float(loaded.channel_mask.sum().item())
        tail_sum = float(loaded.channel_mask[..., -LIGHT_CUT_CHANNEL:].sum().item())
        assert abs(mask_sum - float(3 + LIGHT_AUG_CHANNEL - LIGHT_CUT_CHANNEL)) < 1e-6
        assert abs(tail_sum) < 1e-6
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("[PASS] test_state_save_load_fixed_buffers")


def test_param_count_log():
    """仅打印参数计数供人读, 不断言 (确认量级合理)。"""
    net = PDLTSLightNetwork(**SMALL_NET_KWARGS)  # small kwargs 下的总参数
    total = 0
    trainable = 0
    for name, p in net.named_parameters():
        n = int(np.prod(p.shape))
        total += n
        if not (name.endswith("mask") or name.endswith("is_inited")
                or name.endswith("running_mean") or name.endswith("running_var")
                or name == "channel_mask"):
            trainable += n
    print(f"[INFO] PDLTSLightNetwork(small) total_params={total}, "
          f"trainable_params={trainable}")

    # 再打一组真实 Light 规格
    net2 = PDLTSLightNetwork()  # full default
    total2 = sum(int(np.prod(p.shape)) for _, p in net2.named_parameters())
    print(f"[INFO] PDLTSLightNetwork(full-light) total_params={total2}")
    print("[PASS] test_param_count_log  (no assert, informational only)")


def test_default_full_light_forward():
    """用完整 Light 规格 (num_neighbors=32) 跑一次 forward, 确认能在较大 patch 下工作。

    此测试使用 patch_size=128 检查前向形状。
    """
    net = PDLTSLightNetwork()  # full default Light
    B, N = 1, 128
    x = jt.randn(B, N, 3)
    denoised, ldj, loss_d = net(x)
    assert tuple(denoised.shape) == (B, N, 3)
    print(f"[PASS] test_default_full_light_forward  patch={N} with num_neighbors=32")


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("Light network smoke test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    test_forward_shape_and_dtype()
    test_injection_feature_shapes()
    test_unit_coupling_shape()
    test_grad_flow()
    test_fbm_produces_nonidentity()
    test_eval_determinism()
    test_fbm_channel_mask_values()
    test_injected_flow_roundtrip()
    test_state_save_load_fixed_buffers()
    test_param_count_log()
    test_default_full_light_forward()
    print("=" * 60)
    print("ALL PDLTS LIGHT NETWORK SMOKE TESTS PASSED")
    print("=" * 60)
