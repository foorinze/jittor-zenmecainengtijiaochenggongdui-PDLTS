"""Post-stitch candidate selection smoke tests.

Scope:
  - default-off identity behavior;
  - full-cloud candidate pool and hard top-N shape/budget guards;
  - tiny selector gradient path, without touching patch-local training.
"""

import os
import sys
import tempfile
import json

import jittor as jt
import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_STARTER)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
_SCRIPTS_SHARED = os.path.join(_ROOT, "scripts", "shared")
if _SCRIPTS_SHARED not in sys.path:
    sys.path.insert(0, _SCRIPTS_SHARED)

from src.model.pdlts_light.postselect import (  # noqa: E402
    BlindCandidateScorer,
    PostSelectConfig,
    apply_postselect,
    base_preserve_learned_replace_indices,
    base_preserve_replace_indices,
    candidate_features_jt,
    full_cloud_select,
    generate_candidates,
    identity_scores,
)
from src.system.pdlts_light import PDLTSLightPredictSystem  # noqa: E402
from src.system.pdlts_light import _infer_output_version as train_infer  # noqa: E402
from output_layout import infer_output_version as eval_infer  # noqa: E402


def _cloud(n=128, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 3)).astype(np.float32) * 0.1)


def test_generate_candidates_contains_base_prefix_and_ratio():
    base = _cloud(128, seed=1)
    cand, info = generate_candidates(
        base,
        noisy=base,
        mode="pca_midpoint",
        ratio=1.5,
        seed=7,
    )
    assert cand.shape == (192, 3), cand.shape
    assert np.allclose(cand[:128], base)
    assert np.isfinite(cand).all()
    assert info["base_prefix"] is True
    assert info["num_extra"] == 64
    print("[PASS] test_generate_candidates_contains_base_prefix_and_ratio")


def test_generate_candidates_local_knn_midpoint_is_real_mode():
    base = _cloud(128, seed=13)
    cand, info = generate_candidates(
        base,
        noisy=base,
        mode="local_knn_midpoint",
        ratio=1.25,
        seed=7,
        knn_k=4,
    )
    assert cand.shape == (160, 3), cand.shape
    assert np.allclose(cand[:128], base)
    assert info["candidate_generator_impl"] == "local_knn_midpoint"
    assert info["candidate_knn_k"] == 4
    assert info["num_extra"] == 32
    assert np.isfinite(cand).all()
    print("[PASS] test_generate_candidates_local_knn_midpoint_is_real_mode")


def test_generate_candidates_surface_reject_can_remove_extra():
    base = _cloud(64, seed=14)
    cand, info = generate_candidates(
        base,
        noisy=base,
        mode="pca_midpoint",
        ratio=1.5,
        seed=0,
        surface_reject=True,
        reject_abs_threshold=1e-12,
    )
    assert cand.shape == (64, 3), cand.shape
    assert np.allclose(cand, base)
    assert info["extra_before_reject"] == 32
    assert info["extra_after_reject"] == 0
    assert info["extra_reject_rate"] == 1.0
    print("[PASS] test_generate_candidates_surface_reject_can_remove_extra")


def test_full_cloud_select_exact_unique_and_fail_loud():
    cand = _cloud(16, seed=2)
    scores = np.arange(16, dtype=np.float32)
    selected, idx = full_cloud_select(cand, scores, keep_n=8, return_index=True)
    assert selected.shape == (8, 3)
    assert len(set(idx.tolist())) == 8
    assert idx[0] == 15 and idx[-1] == 8

    failed = False
    try:
        full_cloud_select(cand[:4], scores[:4], keep_n=8)
    except ValueError:
        failed = True
    assert failed, "M<N must fail loudly instead of padding duplicate points"
    print("[PASS] test_full_cloud_select_exact_unique_and_fail_loud")


def test_apply_postselect_off_is_identity():
    base = _cloud(64, seed=3)
    noisy = base + 0.01
    out, info = apply_postselect(
        base,
        noisy,
        PostSelectConfig(postselect_mode="off"),
    )
    assert out.shape == base.shape
    assert np.array_equal(out, base)
    assert info["enabled"] is False
    assert info["num_candidates"] == 64
    assert info["coverage_reference"] == "base_stitching"
    print("[PASS] test_apply_postselect_off_is_identity")


def test_apply_postselect_geometric_identity_preserves_base_order():
    base = _cloud(64, seed=4)
    noisy = base + 0.01
    out, info = apply_postselect(
        base,
        noisy,
        PostSelectConfig(
            postselect_mode="geometric",
            candidate_generator_mode="pca_midpoint",
            selector_mode="identity",
            candidate_ratio=1.5,
            seed=11,
        ),
    )
    assert out.shape == base.shape
    assert np.array_equal(out, base)
    assert info["enabled"] is True
    assert info["num_candidates"] == 96
    assert info["keep_n"] == 64
    assert info["selected_unique"] == 64
    assert info["selected_from_base"] == 64
    assert info["coverage_reference"] == "preselect_base_stitching"
    assert info["realized_coverage_recomputed"] is False
    print("[PASS] test_apply_postselect_geometric_identity_preserves_base_order")


def test_apply_postselect_non_identity_changes_output_but_keeps_n():
    x = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    base = np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)
    noisy = base.copy()
    out, info = apply_postselect(
        base,
        noisy,
        PostSelectConfig(
            postselect_mode="geometric",
            candidate_generator_mode="pca_midpoint",
            selector_mode="x_desc",
            candidate_ratio=1.5,
            seed=0,
        ),
    )
    assert out.shape == base.shape
    assert not np.array_equal(out, base), "non-identity selector should change output order/set"
    assert np.unique(out, axis=0).shape[0] == 64
    assert info["selected_unique"] == 64
    assert info["keep_n"] == 64
    assert info["selected_from_base"] < 64
    print("[PASS] test_apply_postselect_non_identity_changes_output_but_keeps_n")


def test_apply_postselect_fps_selector_keeps_unique_candidate_subset():
    base = _cloud(64, seed=12)
    noisy = base + 0.01
    out, info = apply_postselect(
        base,
        noisy,
        PostSelectConfig(
            postselect_mode="geometric",
            candidate_generator_mode="pca_midpoint",
            selector_mode="fps",
            candidate_ratio=1.5,
            seed=3,
        ),
    )
    assert out.shape == base.shape
    assert np.unique(out, axis=0).shape[0] == 64
    assert info["selected_unique"] == 64
    assert info["keep_n"] == 64
    assert info["selected_from_base"] < 64
    assert info["fps_backend"] == "exact_greedy"
    print("[PASS] test_apply_postselect_fps_selector_keeps_unique_candidate_subset")


def test_base_preserve_replace_is_k_limited_and_zero_leakage_geometry():
    base = _cloud(100, seed=15)
    noisy = base + 0.01
    out, info = apply_postselect(
        base,
        noisy,
        PostSelectConfig(
            postselect_mode="candidate_select",
            candidate_generator_mode="local_knn_midpoint",
            selector_mode="base_preserve_replace",
            candidate_ratio=1.5,
            candidate_surface_reject=True,
            candidate_reject_abs_threshold=0.0,
            replace_ratio=0.02,
            seed=4,
        ),
    )
    assert out.shape == base.shape
    assert np.unique(out, axis=0).shape[0] == 100
    assert info["replace_count_actual"] == 2
    assert info["selected_from_base"] == 98
    assert info["drop_mode"] == "densest_base_by_knn_spacing"
    assert info["extra_select_mode"] == "deterministic_fps"
    assert info["refine_mode"] == "off"
    print("[PASS] test_base_preserve_replace_is_k_limited_and_zero_leakage_geometry")


def test_base_preserve_learned_replace_extra_topk_only():
    base = _cloud(20, seed=16)
    extra = _cloud(10, seed=17) + 1.0
    candidates = np.concatenate([base, extra], axis=0)
    scores = np.arange(10, dtype=np.float32)
    idx, info = base_preserve_learned_replace_indices(
        candidates,
        keep_n=20,
        extra_scores=scores,
        replace_ratio=0.10,
    )
    assert idx.shape == (20,)
    assert np.unique(idx).shape[0] == 20
    assert info["replace_count_actual"] == 2
    assert int((idx < 20).sum()) == 18
    assert set(idx[idx >= 20].tolist()) == {28, 29}
    print("[PASS] test_base_preserve_learned_replace_extra_topk_only")


def test_project_local_refine_keeps_shape_and_marks_manifest_info():
    base = _cloud(64, seed=18)
    noisy = base + 0.01
    out, info = apply_postselect(
        base,
        noisy,
        PostSelectConfig(
            postselect_mode="candidate_select",
            candidate_generator_mode="local_knn_midpoint",
            selector_mode="base_preserve_replace",
            refine_mode="project_local",
            candidate_ratio=1.25,
            replace_ratio=0.05,
            seed=5,
        ),
    )
    assert out.shape == base.shape
    assert np.isfinite(out).all()
    assert np.unique(out, axis=0).shape[0] == 64
    assert info["refine_mode"] == "project_local"
    assert info["refine_basis"] == "base_only_local_pca"
    assert info["projected_points"] == 64
    print("[PASS] test_project_local_refine_keeps_shape_and_marks_manifest_info")


def test_predict_system_real_init_defaults_postselect_off():
    base = _cloud(32, seed=8)
    noisy = base + 0.01
    with tempfile.TemporaryDirectory() as tmpdir:
        system = PDLTSLightPredictSystem(
            dataset_module=None,
            model=object(),
            writer=None,
            run_id="postselect_real_init_default_off",
            run_tag="a_final_postselect_smoke",
            run_output_root=tmpdir,
        )
        assert system.postselect_mode == "off"
        assert system.candidate_generator_mode == "off"
        assert system.selector_mode == "identity"
        assert system.refine_mode == "off"
        out, info = system._apply_postselect(base, noisy)
        assert np.array_equal(out, base)
        assert info["enabled"] is False
        assert "a_final" in system.run_dir.replace("\\", "/")

        system._sample_records = [{
            "sample_path": "dummy",
            "n_missing": 0,
            "missing_ratio": 0.0,
            "seed_k_used": 12,
            "level": "L1",
            "postselect_enabled": True,
        }]
        system._finalize_manifest()
        with open(system.manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        summary = manifest["summary"]
        assert summary["postselect_enabled_samples"] == 1
        assert summary["postselect_coverage_reference"] == "preselect_base_stitching"
        assert summary["postselect_realized_coverage_recomputed"] is False
        assert "pre-select base" in summary["postselect_coverage_note"]
    print("[PASS] test_predict_system_real_init_defaults_postselect_off")


def test_predict_system_postselect_hook_default_off_and_geometric_identity():
    base = _cloud(32, seed=9)
    noisy = base + 0.01
    system = PDLTSLightPredictSystem.__new__(PDLTSLightPredictSystem)

    system.postselect_mode = "off"
    system.candidate_generator_mode = "off"
    system.selector_mode = "identity"
    system.refine_mode = "off"
    system.candidate_ratio = 1.5
    system.postselect_seed = 0
    out, info = system._apply_postselect(base, noisy)
    assert np.array_equal(out, base)
    assert info["enabled"] is False

    system.postselect_mode = "geometric"
    system.candidate_generator_mode = "pca_midpoint"
    system.selector_mode = "identity"
    out, info = system._apply_postselect(base, noisy)
    assert np.array_equal(out, base)
    assert info["enabled"] is True
    assert info["num_candidates"] == 48
    assert info["selected_from_base"] == 32
    print("[PASS] test_predict_system_postselect_hook_default_off_and_geometric_identity")


def test_identity_scores_keep_first_n_order():
    scores = identity_scores(num_candidates=10, keep_n=6)
    cand = np.arange(30, dtype=np.float32).reshape(10, 3)
    selected, idx = full_cloud_select(cand, scores, keep_n=6, return_index=True)
    assert idx.tolist() == [0, 1, 2, 3, 4, 5], idx.tolist()
    assert np.array_equal(selected, cand[:6])
    print("[PASS] test_identity_scores_keep_first_n_order")


def test_blind_candidate_scorer_has_nonzero_grad():
    jt.flags.use_cuda = 0
    rng = np.random.default_rng(5)
    cand_np = rng.standard_normal((96, 3)).astype(np.float32)
    cand = jt.array(cand_np)
    features = candidate_features_jt(cand)
    scorer = BlindCandidateScorer(in_channels=features.shape[-1], hidden_channels=16)
    scores = scorer(features)
    weights = jt.array(np.linspace(0.1, 1.0, 96).astype(np.float32))
    loss = (scores * weights).mean()
    params = [p for name, p in scorer.named_parameters()
              if name.endswith("weight") or name.endswith("bias")]
    grads = jt.grad(loss, params)
    grad_sum = 0.0
    for g in grads:
        if g is not None:
            grad_sum += float(jt.abs(g).sum().item())
    assert grad_sum > 1e-12, f"selector scorer grad is zero: {grad_sum}"
    print(f"[PASS] test_blind_candidate_scorer_has_nonzero_grad grad_sum={grad_sum:.3e}")


def test_public_stage_inference_not_unversioned():
    cases = [
        "20260904_a_final_postselect_smoke",
        "20260904_b_final_postselect_smoke_predict",
    ]
    for name in cases:
        expected = "a_final" if "a_final" in name else "b_final"
        assert train_infer(name) == expected
        assert eval_infer(name) == expected
    print("[PASS] test_public_stage_inference_not_unversioned")


if __name__ == "__main__":
    print("=" * 60)
    print("Post-stitch candidate selection smoke")
    print("=" * 60)
    test_generate_candidates_contains_base_prefix_and_ratio()
    test_generate_candidates_local_knn_midpoint_is_real_mode()
    test_generate_candidates_surface_reject_can_remove_extra()
    test_full_cloud_select_exact_unique_and_fail_loud()
    test_apply_postselect_off_is_identity()
    test_apply_postselect_geometric_identity_preserves_base_order()
    test_apply_postselect_non_identity_changes_output_but_keeps_n()
    test_apply_postselect_fps_selector_keeps_unique_candidate_subset()
    test_base_preserve_replace_is_k_limited_and_zero_leakage_geometry()
    test_base_preserve_learned_replace_extra_topk_only()
    test_project_local_refine_keeps_shape_and_marks_manifest_info()
    test_predict_system_real_init_defaults_postselect_off()
    test_predict_system_postselect_hook_default_off_and_geometric_identity()
    test_identity_scores_keep_first_n_order()
    test_blind_candidate_scorer_has_nonzero_grad()
    test_public_stage_inference_not_unversioned()
    print("=" * 60)
    print("ALL POST-STITCH CANDIDATE SELECTION TESTS PASSED")
    print("=" * 60)
