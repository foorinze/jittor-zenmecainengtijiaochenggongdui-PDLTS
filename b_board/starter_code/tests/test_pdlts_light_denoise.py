"""PDLTS Light 推理 / patch stitching 单测。

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_pdlts_light_denoise

验证内容:
    A. patch_denoise (在归一化空间)
        A1. 输出 shape 严格 == 输入 (非整数倍 N 也成立)
        A2. 输出 dtype float32
        A3. 注入非零权重后 denoised != noisy
        A4. 输出无 NaN / Inf
        A5. 输出点顺序: identity network forward 下 patch_denoise(x) ≈ x (逐点)
        A6. N < patch_size 时 effective fallback, 不炸
        A7. coverage gap: N 接近 patch_size 时漏点用 noisy 回填 + 发 warn,
            输出点数不变, 漏点值 == noisy 原值
    B. normalize / denormalize 往返
    C. denoise_full_cloud
        C1. 端到端 (normalize + patch + denormalize) 尺度回到输入
        C2. identity network 下 denoise_full_cloud(noisy) ≈ noisy
    D. PDLTSLightWriter
        D1. test_noisy 路径正确截出 shapenet/<cls>/<id>
        D2. Windows \\ 路径能截 (精确输出路径断言)
        D3. 拒绝写回 dataset/
        D4. 拒绝 shape 不匹配 noisy 的 prediction
        D5. 找不到 shapenet/ 时 fail-loudly (去 fallback)
    E. predict_step 端到端小云
"""

import sys
import os
import tempfile
import shutil
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt
from jittor import nn

from src.model.parse import get_model
from src.model.pdlts_light.denoise import (
    patch_denoise,
    denoise_full_cloud,
    normalize_unit_sphere,
    denormalize_unit_sphere,
)
from src.model.pdlts_light.inn import AffineCoupling
from src.system.pdlts_light import PDLTSLightWriter
from src.data.asset import Asset


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


SMALL_MODEL_CFG = {
    "__target__": "PDLTSLight",
    "pc_channel": 3,
    "aug_channel": 48,
    "n_injector": 12,
    "cut_channel": 24,
    "nflow_module": 12,
    "num_neighbors": 8,
    "mlgc_hidden": 32,
    "coupling_hidden": 32,
    "log_scale_clamp": 0.1,
}


def _make_model_and_inject():
    cfg = dict(SMALL_MODEL_CFG)
    model = get_model(model_config=cfg, transform_config={})
    np.random.seed(31)
    for mod in model.network.flow_assemblies:
        for layer in mod.chain:
            if isinstance(layer, AffineCoupling):
                last = layer.net[-1]
                last.weight = jt.array(
                    np.random.randn(*last.weight.shape).astype(np.float32) * 0.05
                )
                last.bias = jt.array(
                    np.random.randn(*last.bias.shape).astype(np.float32) * 0.05
                )
    dummy = jt.randn(1, 128, 3)
    _ = model.network(dummy)
    return model


class IdentityNetwork(nn.Module):
    """Identity dummy: 返回 (patch, ldj=0, loss_d=0).

    用于严格验证 patch_denoise 的 stitching 顺序 (回归测试建议):
    network 透传 => patch_denoise(x) 应逐点 ≈ x.
    """

    def execute(self, x):
        B = x.shape[0]
        ldj = jt.zeros((B,))
        loss_d = jt.zeros((1,))
        return x, ldj, loss_d


# ============================================================================
# A. patch_denoise
# ============================================================================

def test_patch_denoise_shape_preservation():
    model = _make_model_and_inject()
    # 避免 N 太接近 patch_size 触发 coverage gap; 覆盖不足时会记录
    # warn + 回填, 这里测 shape 保持, 用 N >= 2 * patch 保证零漏点.
    for N in [128, 256, 333, 512]:
        noisy = jt.randn(N, 3) * 0.3
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denoised = patch_denoise(
                model.network, noisy,
                patch_size=32, seed_k=3, seed_k_alpha=2,
            )
        assert denoised.shape == noisy.shape, (
            f"shape mismatch at N={N}: in={noisy.shape}, out={denoised.shape}"
        )
    print("[PASS] test_patch_denoise_shape_preservation")


def test_patch_denoise_dtype():
    model = _make_model_and_inject()
    noisy = jt.randn(256, 3) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised = patch_denoise(
            model.network, noisy, patch_size=32, seed_k=3, seed_k_alpha=2,
        )
    assert denoised.dtype == "float32"
    assert denoised.numpy().dtype == np.float32
    print(f"[PASS] test_patch_denoise_dtype  dtype={denoised.dtype}")


def test_patch_denoise_nontrivial():
    model = _make_model_and_inject()
    noisy = jt.randn(256, 3) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised = patch_denoise(
            model.network, noisy, patch_size=32, seed_k=3, seed_k_alpha=2,
        )
    diff = float(jt.abs(denoised - noisy).max().item())
    assert diff > 1e-3, f"too close: max_diff={diff:.2e}"
    assert np.isfinite(diff)
    print(f"[PASS] test_patch_denoise_nontrivial  max_diff={diff:.4e}")


def test_patch_denoise_no_nan_inf():
    model = _make_model_and_inject()
    noisy = jt.randn(256, 3) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised = patch_denoise(
            model.network, noisy, patch_size=32, seed_k=3, seed_k_alpha=2,
        )
    assert np.isfinite(denoised.numpy()).all()
    print("[PASS] test_patch_denoise_no_nan_inf")


def test_patch_denoise_identity_network_preserves_input():
    """回归测试: identity network + patch_denoise 应该逐点等于输入."""
    ident = IdentityNetwork()
    N = 256
    noisy = jt.randn(N, 3) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = patch_denoise(
            ident, noisy, patch_size=32, seed_k=3, seed_k_alpha=2,
        )
    err = float(jt.abs(out - noisy).max().item())
    # identity 下 stitching 唯一的误差源是 float 精度, 应该 < 1e-5
    assert err < 1e-4, f"identity stitching err = {err:.2e}"
    print(f"[PASS] test_patch_denoise_identity_network_preserves_input  err={err:.2e}")


def test_patch_denoise_small_N_fallback():
    model = _make_model_and_inject()
    N = 32
    noisy = jt.randn(N, 3) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised = patch_denoise(
            model.network, noisy,
            patch_size=64, seed_k=3, seed_k_alpha=2,
        )
    assert denoised.shape == (N, 3)
    assert np.isfinite(denoised.numpy()).all()
    print(f"[PASS] test_patch_denoise_small_N_fallback  N={N} effective<=64")


def test_patch_denoise_coverage_gap_fallback():
    """N 接近 patch_size 时 FPS+KNN 会漏点, 测漏点用 noisy 回填."""
    ident = IdentityNetwork()
    # 故意制造容易漏点的参数: N=128, patch_size=64, seed_k=3
    # (这也是 WSL 实测失败的 setup)
    N = 128
    noisy = jt.randn(N, 3) * 0.3
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        denoised = patch_denoise(
            ident, noisy,
            patch_size=64, seed_k=3, seed_k_alpha=2,
        )
        # 要么没漏 (ceil 后可能刚好全覆盖), 要么漏了且发 warn
        coverage_warns = [x for x in w if "coverage gap" in str(x.message)]

    # 即使漏点, shape 仍然不变 + 值有限
    assert denoised.shape == (N, 3)
    assert np.isfinite(denoised.numpy()).all()
    # identity 网络下, 无论是否漏点, patch_denoise(x) 都应逐点 = x
    # (覆盖点 denoised[p] = x[p] 因为 identity; 漏点 denoised[p] = noisy[p] = x[p])
    err = float(jt.abs(denoised - noisy).max().item())
    assert err < 1e-4, f"identity path_denoise err = {err:.2e}"
    if coverage_warns:
        print(f"[PASS] test_patch_denoise_coverage_gap_fallback  "
              f"coverage gap happened and was filled with noisy; warn emitted")
    else:
        print(f"[PASS] test_patch_denoise_coverage_gap_fallback  "
              f"(no coverage gap at K=ceil; stitching still identity-exact)")


def test_patch_denoise_returns_coverage_info():
    """return_coverage=True 时应返回 (var, dict) tuple, dict 含规定字段."""
    ident = IdentityNetwork()
    N = 256
    noisy = jt.randn(N, 3) * 0.3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = patch_denoise(
            ident, noisy,
            patch_size=32, seed_k=3, seed_k_alpha=2,
            return_coverage=True,
        )
    assert isinstance(result, tuple) and len(result) == 2
    denoised, info = result
    assert denoised.shape == (N, 3)
    # info 必含字段
    for key in ("n_missing", "missing_ratio", "K", "effective_patch_size",
                "seed_k", "N"):
        assert key in info, f"coverage_info missing key '{key}'"
    # 值合理
    assert info["N"] == N
    assert info["seed_k"] == 3
    assert info["effective_patch_size"] == 32
    assert info["K"] >= 1
    assert info["n_missing"] >= 0
    assert 0.0 <= info["missing_ratio"] <= 1.0
    assert abs(info["missing_ratio"] - info["n_missing"] / N) < 1e-9
    print(f"[PASS] test_patch_denoise_returns_coverage_info  "
          f"n_missing={info['n_missing']}, K={info['K']}")


def test_denoise_full_cloud_returns_coverage_info():
    """denoise_full_cloud(return_coverage=True) 传递 coverage_info 到顶层."""
    ident = IdentityNetwork()
    np.random.seed(202)
    noisy_np = np.random.randn(256, 3).astype(np.float32) * 3.0 + 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised_np, info = denoise_full_cloud(
            ident, noisy_np,
            patch_size=32, seed_k=3, seed_k_alpha=2,
            return_coverage=True,
        )
    assert denoised_np.shape == noisy_np.shape
    assert "n_missing" in info and "missing_ratio" in info
    print(f"[PASS] test_denoise_full_cloud_returns_coverage_info  "
          f"n_missing={info['n_missing']}")


# ============================================================================
# B. normalize / denormalize 往返
# ============================================================================

def test_normalize_roundtrip():
    np.random.seed(99)
    x = np.random.randn(200, 3).astype(np.float32) * 5.0 + 10.0
    normed, center, scale = normalize_unit_sphere(x)
    back = denormalize_unit_sphere(normed, center, scale)
    err = float(np.abs(back - x).max())
    assert err < 1e-4, f"normalize roundtrip err = {err}"
    print(f"[PASS] test_normalize_roundtrip  err={err:.2e}")


def test_normalize_inside_unit_sphere():
    np.random.seed(99)
    x = np.random.randn(200, 3).astype(np.float32) * 5.0 + 10.0
    normed, _, _ = normalize_unit_sphere(x)
    max_norm = float(np.linalg.norm(normed, axis=-1).max())
    assert max_norm <= 1.0 + 1e-5, f"max L2 norm = {max_norm}, should be <= 1"
    print(f"[PASS] test_normalize_inside_unit_sphere  max_norm={max_norm:.4f}")


# ============================================================================
# C. denoise_full_cloud
# ============================================================================

def test_denoise_full_cloud_end_to_end():
    """完整入口: normalize + patch + denormalize, 输出有限 + shape 不变.

    注: 不再断言 'output center ~= input center' 这种几何守恒 —— 未训练的网络
    本来就会大幅改变点分布. identity-network 路径 (C2) 专门验证尺度恢复.
    """
    model = _make_model_and_inject()
    np.random.seed(100)
    noisy_np = np.random.randn(256, 3).astype(np.float32) * 3.0 + 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        denoised_np = denoise_full_cloud(
            model.network, noisy_np,
            patch_size=32, seed_k=3, seed_k_alpha=2,
        )
    assert denoised_np.shape == noisy_np.shape
    assert denoised_np.dtype == np.float32
    assert np.isfinite(denoised_np).all()
    print(f"[PASS] test_denoise_full_cloud_end_to_end  "
          f"shape={denoised_np.shape}, finite")


def test_denoise_full_cloud_identity():
    """回归测试: identity network + denoise_full_cloud 应近似恒等.

    唯一误差来自 normalize->denormalize 的 float 精度损失 (非单位球数据).
    """
    ident = IdentityNetwork()
    np.random.seed(101)
    noisy_np = np.random.randn(256, 3).astype(np.float32) * 3.0 + 5.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = denoise_full_cloud(
            ident, noisy_np,
            patch_size=32, seed_k=3, seed_k_alpha=2,
        )
    err = float(np.abs(out - noisy_np).max())
    # normalize+denormalize 的浮点损失预期 < 1e-3 对 scale~3 的输入
    assert err < 1e-3, f"identity full_cloud err = {err:.2e}"
    print(f"[PASS] test_denoise_full_cloud_identity  err={err:.2e}")


# ============================================================================
# D. PDLTSLightWriter
# ============================================================================

def test_writer_path_extraction_test_noisy():
    tmp = tempfile.mkdtemp(prefix="denoise_writer_writer_")
    try:
        writer = PDLTSLightWriter(save_dir=tmp, save_name="denoised")
        asset = Asset(
            path="../dataset/test_noisy/shapenet/04401088/c1c23c7a/noisy.npy",
            cls="shapenet",
        )
        asset.sampled_vertices_noisy = np.random.randn(100, 3).astype(np.float32)
        prediction = [{"pc_denoised": np.random.randn(100, 3).astype(np.float32)}]
        batch = {"asset": [asset], "pc_noisy": None}
        writer.write(batch, prediction)
        expected = os.path.join(tmp, "shapenet", "04401088", "c1c23c7a", "denoised.npy")
        assert os.path.exists(expected), f"missing {expected}"
        loaded = np.load(expected)
        assert loaded.shape == (100, 3)
        assert loaded.dtype == np.float32
        print("[PASS] test_writer_path_extraction_test_noisy")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_writer_path_extraction_windows_sep():
    """Windows \\ 路径必须被正规化后精确写到 shapenet/<cls>/<id>/denoised.npy."""
    tmp = tempfile.mkdtemp(prefix="denoise_writer_writer_win_")
    try:
        writer = PDLTSLightWriter(save_dir=tmp, save_name="denoised")
        asset = Asset(
            path=r"..\dataset\test_noisy\shapenet\04401088\c1c23c7a\noisy.npy",
            cls="shapenet",
        )
        asset.sampled_vertices_noisy = np.random.randn(50, 3).astype(np.float32)
        prediction = [{"pc_denoised": np.random.randn(50, 3).astype(np.float32)}]
        batch = {"asset": [asset], "pc_noisy": None}
        writer.write(batch, prediction)
        # 精确路径断言 (而不是只检查 "shapenet" 在某处)
        expected = os.path.join(tmp, "shapenet", "04401088", "c1c23c7a", "denoised.npy")
        assert os.path.exists(expected), f"expected {expected}, not found"
        loaded = np.load(expected)
        assert loaded.shape == (50, 3)
        print("[PASS] test_writer_path_extraction_windows_sep")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_writer_refuses_to_write_into_dataset():
    dataset_root = os.path.abspath(
        os.path.join(_STARTER, "..", "dataset")
    )
    bad_save_dir = os.path.join(dataset_root, "pdlts_accidental_output")
    writer = PDLTSLightWriter(save_dir=bad_save_dir, save_name="denoised")
    asset = Asset(
        path="../dataset/test_noisy/shapenet/04401088/c1c23c7a/noisy.npy",
        cls="shapenet",
    )
    asset.sampled_vertices_noisy = np.random.randn(20, 3).astype(np.float32)
    prediction = [{"pc_denoised": np.random.randn(20, 3).astype(np.float32)}]
    batch = {"asset": [asset], "pc_noisy": None}
    try:
        writer.write(batch, prediction)
    except AssertionError as e:
        assert "dataset" in str(e).lower()
        print("[PASS] test_writer_refuses_to_write_into_dataset")
        return
    raise AssertionError("writer did NOT refuse to write into dataset/")


def test_writer_shape_guard():
    tmp = tempfile.mkdtemp(prefix="denoise_writer_writer_guard_")
    try:
        writer = PDLTSLightWriter(save_dir=tmp, save_name="denoised")
        asset = Asset(
            path="../dataset/test_noisy/shapenet/04401088/c1c23c7a/noisy.npy",
            cls="shapenet",
        )
        asset.sampled_vertices_noisy = np.random.randn(100, 3).astype(np.float32)
        prediction = [{"pc_denoised": np.random.randn(90, 3).astype(np.float32)}]
        batch = {"asset": [asset], "pc_noisy": None}
        try:
            writer.write(batch, prediction)
        except AssertionError:
            print("[PASS] test_writer_shape_guard")
            return
        raise AssertionError("writer failed to reject shape-mismatched prediction")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_writer_rejects_path_without_shapenet_anchor():
    """回归测试: 找不到 'shapenet/' 锚点时应 fail-loudly, 不走 fallback."""
    tmp = tempfile.mkdtemp(prefix="denoise_writer_writer_bad_")
    try:
        writer = PDLTSLightWriter(save_dir=tmp, save_name="denoised")
        # 故意构造没有 shapenet 锚点的路径
        asset = Asset(
            path="../dataset/test_noisy/OTHER_ROOT/04401088/c1c/noisy.npy",
            cls="shapenet",
        )
        asset.sampled_vertices_noisy = np.random.randn(20, 3).astype(np.float32)
        prediction = [{"pc_denoised": np.random.randn(20, 3).astype(np.float32)}]
        batch = {"asset": [asset], "pc_noisy": None}
        try:
            writer.write(batch, prediction)
        except AssertionError as e:
            assert "shapenet" in str(e).lower(), \
                f"expected shapenet anchor error, got: {e}"
            print("[PASS] test_writer_rejects_path_without_shapenet_anchor")
            return
        raise AssertionError("writer failed to reject path without shapenet anchor")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ============================================================================
# E. predict_step 端到端
# ============================================================================

def test_predict_step_end_to_end_small():
    """predict_step 端到端, 只断言 shape / dtype / finite.

    同 test_denoise_full_cloud_end_to_end: 未训练网络不应有尺度守恒预期.
    """
    model = _make_model_and_inject()
    model.set_predict(True)

    N = 256
    batch = {"pc_noisy": jt.randn(1, N, 3) * 2.0 + 5.0}

    model.model_config["patch_size"] = 32
    model.model_config["predict_seed_k"] = 3
    model.model_config["predict_seed_k_alpha"] = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = model.predict_step(batch)
    assert len(out) == 1
    d = out[0]
    assert "pc_denoised" in d
    denoised = d["pc_denoised"]
    assert isinstance(denoised, np.ndarray)
    assert denoised.shape == (N, 3)
    assert denoised.dtype == np.float32
    assert np.isfinite(denoised).all()
    print(f"[PASS] test_predict_step_end_to_end_small  shape={denoised.shape}")


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("patch stitching / denoise smoke test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    # A. patch_denoise
    test_patch_denoise_shape_preservation()
    test_patch_denoise_dtype()
    test_patch_denoise_nontrivial()
    test_patch_denoise_no_nan_inf()
    test_patch_denoise_identity_network_preserves_input()
    test_patch_denoise_small_N_fallback()
    test_patch_denoise_coverage_gap_fallback()
    test_patch_denoise_returns_coverage_info()
    # B. normalize
    test_normalize_roundtrip()
    test_normalize_inside_unit_sphere()
    # C. denoise_full_cloud
    test_denoise_full_cloud_end_to_end()
    test_denoise_full_cloud_identity()
    test_denoise_full_cloud_returns_coverage_info()
    # D. writer
    test_writer_path_extraction_test_noisy()
    test_writer_path_extraction_windows_sep()
    test_writer_refuses_to_write_into_dataset()
    test_writer_shape_guard()
    test_writer_rejects_path_without_shapenet_anchor()
    # E. predict_step
    test_predict_step_end_to_end_small()
    print("=" * 60)
    print("ALL PDLTS LIGHT DENOISE TESTS PASSED")
    print("=" * 60)
