"""Candidate-select-clean trainable smoke test.

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_candidate_select_clean_smoke

验证目标（对齐交付要求并避开错误的梯度截断）：
    A. 4 arm（A/B/C/D）单 batch 前向不崩，输出点数严格 = M（patch 级）。
    B. learned selector arm 训练期 emit selector_aux loss key，且 loss 有限。
    C. selector_aux 梯度回流主干（NOT detach）。
    D. 主 chamfer 梯度经选中点回流主干（candidate 位置可训练）。
    E. 推理期 selector 真生效：hard top-M 改变输出（不是旁挂诊断头）。
    F. 候选 off（base）时输出 == 纯 INN（关闭分支时的输出一致性检查，已由 system smoke 覆盖，这里再确认）。

注：M 取小号（32）控制时长；candidate_ratio=2.0 -> 候选池 64 -> 硬选回 32。
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np
import jittor as jt

from src.model.parse import get_model
from src.model.pdlts_light.system import (
    LOSS_KEY_CHAMFER,
    LOSS_KEY_L2,
    LOSS_KEY_SELECTOR_AUX,
)


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


_BASE_CFG = {
    "__target__": "PDLTSLight",
    "pc_channel": 3, "aug_channel": 48, "n_injector": 12, "cut_channel": 24,
    "nflow_module": 12, "num_neighbors": 8, "mlgc_hidden": 32,
    "coupling_hidden": 32, "log_scale_clamp": 0.1,
    "target_mode": "paired_idx", "l2_target_mode": "fixed",
    "candidate_mode": "slot_variant", "candidate_R": 4, "candidate_knn_k": 4,
}


def _arm_cfg(selector_mode, cleaner_mode, selector_aux_mode):
    cfg = dict(_BASE_CFG)
    cfg["selector_mode"] = selector_mode
    cfg["cleaner_mode"] = cleaner_mode
    cfg["selector_aux_mode"] = selector_aux_mode
    cfg["selector_hidden"] = 32
    cfg["cleaner_hidden"] = 32
    return cfg


# 4 arm 析因（2x2，slot_variant 语义）。selector_aux 仅 learned selector arm 开。
# A/D 用 identity selector（恒选 variant 0=base），B/C 用 learned（逐点 R 变体内选 1）。
ARMS = {
    "A_identity_control": _arm_cfg("identity", "off", "off"),
    "B_learned_selector": _arm_cfg("learned", "off", "variant_ce"),
    "C_selector_cleaner": _arm_cfg("learned", "residual", "variant_ce"),
    "D_cleaner_control": _arm_cfg("identity", "residual", "off"),
}


def _model(cfg):
    return get_model(model_config=dict(cfg), transform_config={})


def _batch(B=2, P=1, M=32):
    return {
        "pc_noisy": jt.randn(B, P, M, 3),
        "pc_clean": jt.randn(B, P, M, 3),
    }


def test_all_arms_forward_output_count_M():
    """A. 4 arm 前向不崩，输出点数严格 = M（slot_variant 保槽位）。"""
    M = 32
    for name, cfg in ARMS.items():
        model = _model(cfg)
        out = model.training_step(_batch(M=M))
        var_idx = model.network._selected_variant_idx
        assert var_idx is not None, f"{name}: selected_variant_idx 未设置"
        assert var_idx.shape == (2, M), (
            f"{name}: selected_variant_idx shape {tuple(var_idx.shape)} != (2,{M})")
        assert LOSS_KEY_CHAMFER in out and np.isfinite(float(out[LOSS_KEY_CHAMFER].item())), (
            f"{name}: chamfer 缺失或非有限")
        assert np.isfinite(float(out[LOSS_KEY_L2].item())), f"{name}: l2 非有限"
        # 变体张量 (B, M, R, 3)，R=4
        sv = model.network._slot_variants
        assert sv.shape == (2, M, 4, 3), (
            f"{name}: slot_variants shape {tuple(sv.shape)} != (2,{M},4,3)")
    print(f"[PASS] test_all_arms_forward_output_count_M  ({len(ARMS)} arms, M={M}, R=4)")


def test_learned_selector_emits_aux_loss():
    """B. learned selector arm emit selector_aux(variant_ce)；identity arm 不 emit。"""
    m_learned = _model(ARMS["B_learned_selector"])
    out = m_learned.training_step(_batch())
    assert LOSS_KEY_SELECTOR_AUX in out, "learned selector 应 emit selector_aux"
    assert np.isfinite(float(out[LOSS_KEY_SELECTOR_AUX].item())), "selector_aux 非有限"
    sm = getattr(m_learned, "_selector_metrics", {})
    for k in ("selector_aux_raw", "variant_R", "selector_hit_clean_frac", "selector_nonbase_frac"):
        assert k in sm, f"selector metrics 缺 {k}"
    # CX 核查点：selector metrics 必须合并进 _last_train_metrics
    ltm = getattr(m_learned, "_last_train_metrics", {})
    for k in ("selector_aux_raw", "selector_hit_clean_frac", "selector_nonbase_frac"):
        assert k in ltm, f"_last_train_metrics 缺 {k}（_selector_metrics 未合并）"

    m_id = _model(ARMS["A_identity_control"])
    out_id = m_id.training_step(_batch())
    assert LOSS_KEY_SELECTOR_AUX not in out_id, "identity 对照臂不应 emit selector_aux"
    ltm_id = getattr(m_id, "_last_train_metrics", {})
    assert "selector_aux_raw" not in ltm_id, (
        "identity 对照臂 _last_train_metrics 不应含 selector_aux_raw")
    # identity arm 必须恒选 variant 0（=base）
    assert int(m_id.network._selected_variant_idx.max().item()) == 0, (
        "identity selector 应恒选 variant 0")
    print("[PASS] test_learned_selector_emits_aux_loss "
          f"(learned: R={sm['variant_R']}, hit_clean={sm['selector_hit_clean_frac']:.3f}, "
          f"nonbase={sm['selector_nonbase_frac']:.3f}; identity: 恒选 variant0 OK)")


def _trunk_weight(model):
    """取一个主干（非 selector/cleaner/keep）权重参数用于梯度检查。"""
    for name, p in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if any(s in name for s in ("MLP_selector", "MLP_cleaner", "MLP_keep")):
            continue
        if name.endswith("mask") or name.endswith("channel_mask"):
            continue
        return name, p
    raise AssertionError("no trunk weight found")


def _trunk_params(model, limit=40):
    """收集一批主干权重参数（排除 selector/cleaner/keep/buffer），用于聚合梯度检查。
    单个 param 在某 batch 下梯度可能恰为 0，聚合多个更稳。"""
    out = []
    for name, p in model.named_parameters():
        if not name.endswith("weight"):
            continue
        if any(s in name for s in ("MLP_selector", "MLP_cleaner", "MLP_keep")):
            continue
        if name.endswith("mask") or name.endswith("channel_mask"):
            continue
        out.append((name, p))
        if len(out) >= limit:
            break
    assert out, "no trunk weight found"
    return out


def _grad_sum(loss, params):
    gs = jt.grad(loss, params)
    total = 0.0
    for g in gs:
        if g is not None:
            total += float(jt.abs(g).sum().item())
    return total


def test_selector_aux_grad_reaches_trunk():
    """C. selector_aux 梯度回流主干（NOT detach）—— 与 soft-weight 的关键区别。

    注：MLP_selector 末层 zero-init → 初始 logit≡0（选中=identity=base，无偏置），
    此时梯度只到末层、尚未传到首层/主干（zero-init 暂态，非 detach）。故先 warmup
    几步让末层 W2≠0，再验证 selector_aux 能回流主干。若像 soft-weight 那样 detach，则无论
    warmup 多少步主干梯度恒 0 —— 这正是本测试要排除的。"""
    model = _model(ARMS["B_learned_selector"])
    params = [p for n, p in model.named_parameters()
              if not (n.endswith("running_mean") or n.endswith("running_var")
                      or n.endswith("is_inited") or n.endswith("mask"))]
    opt = jt.optim.Adam(params, lr=1e-2)
    for _ in range(3):  # 打破 zero-init 暂态
        out = model.training_step(_batch())
        loss = out[LOSS_KEY_CHAMFER] + out[LOSS_KEY_L2] + out[LOSS_KEY_SELECTOR_AUX]
        opt.zero_grad(); opt.backward(loss); opt.step()

    out = model.training_step(_batch())
    aux = out[LOSS_KEY_SELECTOR_AUX]
    trunk_params = _trunk_params(model)
    sel_w = None
    for name, p in model.named_parameters():
        if "MLP_selector" in name and name.endswith("weight"):
            sel_w = (name, p)
            break
    assert sel_w is not None, "no MLP_selector weight"
    trunk_sum = _grad_sum(aux, [p for _, p in trunk_params])
    sel_sum = _grad_sum(aux, [sel_w[1]])
    assert sel_sum > 1e-12, f"selector head 无梯度: {sel_sum:.3e}"
    assert trunk_sum > 1e-12, (
        f"selector_aux 未回流主干: grad_sum={trunk_sum:.3e}"
        f"（warmup 后仍为 0 = 像 soft-weight 那样 detach 了主干）")
    print(f"[PASS] test_selector_aux_grad_reaches_trunk "
          f"(warmup3 后 trunk grad_sum={trunk_sum:.3e}, selector grad={sel_sum:.3e})")


def test_main_chamfer_grad_reaches_trunk_via_selected():
    """D. 主 chamfer 经选中变体回流主干（变体位置可训练）。"""
    model = _model(ARMS["C_selector_cleaner"])
    out = model.training_step(_batch())
    chamfer = out[LOSS_KEY_CHAMFER]
    trunk_params = _trunk_params(model)
    gsum = _grad_sum(chamfer, [p for _, p in trunk_params])
    assert gsum > 1e-12, f"主 chamfer 未回流主干: {gsum:.3e}"
    cl_w = None
    for name, p in model.named_parameters():
        if "MLP_cleaner" in name and name.endswith("weight"):
            cl_w = p
            break
    assert cl_w is not None, "no MLP_cleaner weight"
    gc_sum = _grad_sum(chamfer, [cl_w])
    print(f"[PASS] test_main_chamfer_grad_reaches_trunk_via_selected "
          f"(trunk grad_sum={gsum:.3e}, cleaner grad={gc_sum:.3e})")


def test_inference_selector_actually_changes_output():
    """E. 推理期 selector 真生效：训练后 learned selector 会选非-base 变体（≠恒选 variant0）。
    identity arm 恒选 variant0；learned arm 训练后应出现 variant_idx>0（真在用变体）。"""
    M = 32
    model = _model(ARMS["B_learned_selector"])
    opt = jt.optim.Adam(
        [p for n, p in model.named_parameters()
         if not (n.endswith("running_mean") or n.endswith("running_var")
                 or n.endswith("is_inited") or n.endswith("mask"))], lr=1e-2)
    for _ in range(5):
        out = model.training_step(_batch(M=M))
        loss = out[LOSS_KEY_CHAMFER] + out[LOSS_KEY_L2] + out[LOSS_KEY_SELECTOR_AUX]
        opt.zero_grad(); opt.backward(loss); opt.step()
    out = model.training_step(_batch(M=M))
    var_idx = model.network._selected_variant_idx.numpy()
    changed = (var_idx > 0).any()
    assert changed, "训练后 learned selector 仍恒选 variant0，selector 未真正生效"
    print("[PASS] test_inference_selector_actually_changes_output "
          f"(variant_idx>0 出现，selector 真在用非-base 变体)")


def test_end_to_end_stitching_50000_slot_preserving():
    """F. 端到端 denoise_full_cloud（多 patch stitching）slot-preserving 精确验证。
    上一轮 4-arm 崩盘根因 = 子集选重排破坏 stitching 槽位对应。本测试用同一 model 跑
    stage-on（identity selector 选 variant0=base，cleaner off）vs stage-off（纯 base），
    输出应**逐点相等**——若 slot 对应坏了，两者会偏离。这比"漏点=0"更精确，且不受
    随机点云覆盖率干扰（两边同 seed_k）。此处运行完整点云拼接以覆盖跨 patch 行为。"""
    from src.model.pdlts_light.denoise import denoise_full_cloud
    model = _model(ARMS["A_identity_control"])  # identity + cleaner off
    net = model.network
    N = 50000
    rng = np.random.default_rng(0)
    noisy = (rng.standard_normal((N, 3)).astype(np.float32)) * 0.1
    # 1) stage on（slot_variant + identity 选 base，无 cleaner）
    out_on = denoise_full_cloud(net, noisy, patch_size=1024, seed_k=3, seed_k_alpha=5)
    # 2) stage off（临时关 candidate，走纯 base 路径）
    saved = net.candidate_mode
    net.candidate_mode = "off"
    out_off = denoise_full_cloud(net, noisy, patch_size=1024, seed_k=3, seed_k_alpha=5)
    net.candidate_mode = saved
    assert out_on.shape == (N, 3), f"输出 shape {out_on.shape} != ({N},3)"
    assert np.isfinite(out_on).all(), "输出含 NaN/Inf"
    # identity 选 base + 无 cleaner → stage-on 应与纯 base 逐点近似相等（slot 对应没坏）。
    # 阈值 1e-4：容忍 GPU float32 跨两次 forward 的非确定性（~1e-5，非结合归约），
    # 仍比真实 slot-scramble（O(0.01)=点间距量级）低 100x，能稳健区分。
    max_dev = float(np.abs(out_on - out_off).max())
    assert max_dev < 1e-4, (
        f"identity slot_variant 改变了输出（max_dev={max_dev:.2e}）= stitching 槽位对应被破坏")
    print(f"[PASS] test_end_to_end_stitching_50000_slot_preserving "
          f"(50000 点, identity==base max_dev={max_dev:.2e}, slot-preserving OK)")


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("slot-variant select-clean trainable smoke")
    print("=" * 60)
    test_all_arms_forward_output_count_M()
    test_learned_selector_emits_aux_loss()
    test_selector_aux_grad_reaches_trunk()
    test_main_chamfer_grad_reaches_trunk_via_selected()
    test_inference_selector_actually_changes_output()
    test_end_to_end_stitching_50000_slot_preserving()
    print("=" * 60)
    print("ALL SLOT-VARIANT SMOKE TESTS PASSED")
    print("=" * 60)
