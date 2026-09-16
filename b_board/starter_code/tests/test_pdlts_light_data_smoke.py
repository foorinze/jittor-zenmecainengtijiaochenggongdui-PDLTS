"""PDLTS Light data + ModelSpec smoke 单测。

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_pdlts_light_data_smoke

验证内容:
    1. get_model 能识别 'PDLTSLight'（MAP 注册生效）
    2. process_fn 训练 / 推理两种模式的 dict 结构正确
    3. _chamfer_l2 自身正确性（同点云 chamfer=0，非零点云 > 0）
    4. training_step 能从 batched dict 算 loss dict（两个 key + 都是有限非零）
    5. 端到端：configs/transform/_shared/pdlts_light.yaml + AugmentPatch 产出的 (P=1, M=1024, 3)
       能被 process_fn 正确打包（pc_mix 被忽略）
    6. 端到端真实数据：加载一个 dataset/train 下的 .obj，走完整 transform 链后喂进
       training_step，loss 有限。
    7. predict_transform 是空 augments，不读取网格

注：不启动 DataLoader 多 worker / 不跑完整 PCDataset，完整训练系统另有测试。
    这里只验证"一条数据从 .obj 到 loss 的最小通路"。

注意：中心化语义差异:
    starter AugmentPatch 用的是 "seed_points_t" (clean/noisy 插值 seed) 减法,
    原 PD-LTS denoise.py 用的是 "纯 noisy seed" 减法. 两者有语义差异，
    此处使用插值种子中心，与原版推理的中心定义不同。
"""

import sys
import os
import glob

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np
from omegaconf import OmegaConf

import jittor as jt
from jittor import nn

from src.data.asset import Asset
from src.data.augment import (
    AugmentSample,
    AugmentNormalizePC,
    AugmentAddNoise,
    AugmentPatch,
)
from src.data.datapath import ObjLazyAsset
from src.data.transform import Transform
from src.model.parse import get_model
from src.model.pdlts_light.system import (
    PDLTSLight,
    _chamfer_l2,
    LOSS_KEY_CHAMFER,
    LOSS_KEY_L2,
)


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


# 为了控制单测时长, 用小网络
SMALL_MODEL_CFG = {
    "__target__": "PDLTSLight",
    "pc_channel": 3,
    "aug_channel": 48,
    "n_injector": 12,
    "cut_channel": 24,
    "nflow_module": 12,
    "num_neighbors": 8,      # 小 k 够 smoke
    "mlgc_hidden": 32,
    "coupling_hidden": 32,
    "log_scale_clamp": 0.1,
}


def test_get_model_registers_pdlts_light():
    """get_model('PDLTSLight', ...) 能正确实例化 PDLTSLight."""
    cfg = dict(SMALL_MODEL_CFG)       # copy, get_model 会 del __target__
    model = get_model(model_config=cfg, transform_config={})
    assert isinstance(model, PDLTSLight), f"expected PDLTSLight, got {type(model)}"
    print("[PASS] test_get_model_registers_pdlts_light")


def test_chamfer_l2_identity():
    """_chamfer_l2 对同点云应当 ≈ 0 (每个点自己是最近邻)."""
    B, N = 2, 32
    x = jt.randn(B, N, 3)
    d = _chamfer_l2(x, x)
    val = float(d.item())
    assert val < 1e-6, f"chamfer(x, x) should be ~0, got {val}"
    print(f"[PASS] test_chamfer_l2_identity  value={val:.2e}")


def test_chamfer_l2_nontrivial():
    """两个独立高斯点云的 chamfer 明显 > 0."""
    jt.set_global_seed(7)
    x = jt.randn(2, 32, 3)
    y = jt.randn(2, 32, 3)
    d = _chamfer_l2(x, y)
    val = float(d.item())
    assert val > 0.1, f"chamfer(different clouds) expected > 0.1, got {val}"
    print(f"[PASS] test_chamfer_l2_nontrivial  value={val:.4f}")


def test_process_fn_training_dict_shape():
    """训练模式: process_fn 把 asset.meta['pc_noisy'/'pc_clean'] 转为 float32 array."""
    cfg = dict(SMALL_MODEL_CFG)
    model = get_model(model_config=cfg, transform_config={})
    model.set_predict(False)

    P, M = 1, 64
    asset = Asset()
    asset.meta = {
        "pc_noisy": np.random.randn(P, M, 3).astype(np.float64),    # 故意给 f64
        "pc_clean": np.random.randn(P, M, 3).astype(np.float64),
        "pc_mix":   np.random.randn(P, M, 3).astype(np.float64),    # 应该被忽略
    }

    res = model.process_fn([asset])
    assert len(res) == 1, f"expected 1 dict, got {len(res)}"
    d = res[0]
    assert set(d.keys()) == {"pc_noisy", "pc_clean"}, \
        f"pc_mix should be ignored, got keys={list(d.keys())}"
    assert d["pc_noisy"].dtype == np.float32
    assert d["pc_noisy"].shape == (P, M, 3)
    print(f"[PASS] test_process_fn_training_dict_shape  keys={list(d.keys())}")


def test_process_fn_predict_dict_shape():
    """推理模式: 从 asset.sampled_vertices_noisy 读整云."""
    cfg = dict(SMALL_MODEL_CFG)
    model = get_model(model_config=cfg, transform_config={})
    model.set_predict(True)

    N = 256
    asset = Asset()
    asset.sampled_vertices_noisy = np.random.randn(N, 3).astype(np.float64)

    res = model.process_fn([asset])
    assert len(res) == 1
    d = res[0]
    assert set(d.keys()) == {"pc_noisy"}
    assert d["pc_noisy"].dtype == np.float32
    assert d["pc_noisy"].shape == (N, 3)
    # 带 clean 的情况
    asset2 = Asset()
    asset2.sampled_vertices_noisy = np.random.randn(N, 3).astype(np.float64)
    asset2.sampled_vertices = np.random.randn(N, 3).astype(np.float64)
    res2 = model.process_fn([asset2])
    assert set(res2[0].keys()) == {"pc_noisy", "pc_clean"}
    print("[PASS] test_process_fn_predict_dict_shape")


def test_training_step_synthetic():
    """training_step 用合成 (B, P, M, 3) 能算出两个非零 loss."""
    cfg = dict(SMALL_MODEL_CFG)
    model = get_model(model_config=cfg, transform_config={})

    B, P, M = 2, 1, 64
    batch = {
        "pc_noisy": jt.randn(B, P, M, 3),
        "pc_clean": jt.randn(B, P, M, 3),
    }
    loss_dict = model.training_step(batch)
    assert set(loss_dict.keys()) == {LOSS_KEY_CHAMFER, LOSS_KEY_L2}
    for k, v in loss_dict.items():
        val = float(v.item())
        assert np.isfinite(val), f"loss '{k}' not finite: {val}"
        assert val > 0, f"loss '{k}' should be positive, got {val}"
    print(f"[PASS] test_training_step_synthetic  "
          f"chamfer={float(loss_dict[LOSS_KEY_CHAMFER].item()):.4f}  "
          f"l2={float(loss_dict[LOSS_KEY_L2].item()):.4f}")


def test_transform_pipeline_end_to_end():
    """完整 AugmentSample + NormalizePC + AddNoise + Patch 后产出 (P, M, 3)."""
    # 构造一个合成 mesh (球面三角剖分的近似)
    V = 256
    F = 300
    vertices = np.random.randn(V, 3).astype(np.float32)
    faces = np.stack([
        np.random.randint(0, V, size=F),
        np.random.randint(0, V, size=F),
        np.random.randint(0, V, size=F),
    ], axis=1)
    asset = Asset(vertices=vertices, faces=faces)

    # 手动跑 transform 链
    # 注: AugmentSample 需要有效的 face; 随机的 face 足够 sample 几个顶点
    AugmentSample(num_samples=4096, num_vertex_samples=256).apply(asset)
    AugmentNormalizePC().apply(asset)
    AugmentAddNoise(noise_std_min=0.01, noise_std_max=0.015).apply(asset)
    # patch_size 在本测试里小一点, 用 M=64 避免单测慢; 5c 真实训练走 1024
    AugmentPatch(patch_size=64, num_patches=1, train_cvm_network=False).apply(asset)

    assert asset.meta is not None
    for k in ("pc_noisy", "pc_clean", "pc_mix"):
        assert k in asset.meta, f"missing {k} in asset.meta"
        assert asset.meta[k].shape == (1, 64, 3), \
            f"meta['{k}'] shape {asset.meta[k].shape} != (1, 64, 3)"
    print("[PASS] test_transform_pipeline_end_to_end  meta has pc_noisy/clean/mix shape (1, 64, 3)")


def test_end_to_end_real_obj():
    """从真实 dataset/train/shapenet/... 加载一个 .obj, 走完整 pipeline 到 training_step.

    检查数据处理输出能否被 ModelSpec 正确接收。
    """
    # 找一个真实的 obj 文件 (dataset/train/shapenet/<cls>/<id>/models/model_normalized.obj)
    candidates = glob.glob(os.path.join(
        _STARTER, "..", "dataset", "train", "shapenet", "*", "*", "models", "model_normalized.obj"
    ))
    if len(candidates) == 0:
        print("[SKIP] test_end_to_end_real_obj  no obj file found under dataset/train/")
        return
    obj_path = candidates[0]
    print(f"       using obj: {os.path.relpath(obj_path, _STARTER)}")

    # 1. 加载
    asset = ObjLazyAsset(path=obj_path, cls="shapenet").load()
    assert asset.vertices is not None and asset.faces is not None, \
        "ObjLazyAsset.load failed to populate vertices/faces"

    # 2. transform 链 (对应 configs/transform/_shared/pdlts_light.yaml train_transform)
    AugmentSample(num_samples=4096, num_vertex_samples=256).apply(asset)
    AugmentNormalizePC().apply(asset)
    AugmentAddNoise(noise_std_min=0.01, noise_std_max=0.015).apply(asset)
    # 用小 patch 快速 smoke
    M = 64
    AugmentPatch(patch_size=M, num_patches=1, train_cvm_network=False).apply(asset)

    # 3. process_fn
    cfg = dict(SMALL_MODEL_CFG)
    model = get_model(model_config=cfg, transform_config={})
    model.set_predict(False)
    process_out = model.process_fn([asset])
    assert len(process_out) == 1
    d = process_out[0]
    # 模拟 PCDataset._collate_fn 的 stack: (P, M, 3) -> (B=1, P, M, 3)
    batch = {
        "pc_noisy": jt.array(d["pc_noisy"]).unsqueeze(0),
        "pc_clean": jt.array(d["pc_clean"]).unsqueeze(0),
    }

    # 4. training_step
    loss_dict = model.training_step(batch)
    assert set(loss_dict.keys()) == {LOSS_KEY_CHAMFER, LOSS_KEY_L2}
    for k, v in loss_dict.items():
        val = float(v.item())
        assert np.isfinite(val), f"loss '{k}' not finite: {val}"
    print(f"[PASS] test_end_to_end_real_obj  "
          f"chamfer={float(loss_dict[LOSS_KEY_CHAMFER].item()):.4f}  "
          f"l2={float(loss_dict[LOSS_KEY_L2].item()):.4f}")


def test_predict_transform_is_empty():
    """configs/transform/_shared/pdlts_light.yaml 的 predict_transform.augments 必须是空列表.

    predict 走 noisy.npy, 没有 mesh faces,
    不能复用 validate_transform (AugmentSample 会炸).

    推理切块和拼接由 pdlts_light/denoise.py 实现。
    """
    cfg_path = os.path.join(
        _STARTER, "configs", "transform", "_shared", "pdlts_light.yaml"
    )
    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    assert "predict_transform" in cfg, "missing predict_transform section"
    predict_cfg = cfg["predict_transform"]
    assert predict_cfg.get("augments") == [], \
        f"predict_transform.augments must be empty, got {predict_cfg.get('augments')}"

    # 还要确认 Transform.parse 能处理空 augments
    t = Transform.parse(**predict_cfg)
    # 对一个只有 sampled_vertices_noisy 的 asset 调 apply 不应报错
    asset = Asset()
    asset.sampled_vertices_noisy = np.random.randn(128, 3).astype(np.float32)
    t.apply(asset)  # 空 augments 应无 op
    # asset 不能被篡改
    assert asset.sampled_vertices_noisy.shape == (128, 3)
    print("[PASS] test_predict_transform_is_empty")


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("data + ModelSpec smoke test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    test_get_model_registers_pdlts_light()
    test_chamfer_l2_identity()
    test_chamfer_l2_nontrivial()
    test_process_fn_training_dict_shape()
    test_process_fn_predict_dict_shape()
    test_training_step_synthetic()
    test_transform_pipeline_end_to_end()
    test_end_to_end_real_obj()
    test_predict_transform_is_empty()
    print("=" * 60)
    print("ALL PDLTS LIGHT DATA SMOKE TESTS PASSED")
    print("=" * 60)
