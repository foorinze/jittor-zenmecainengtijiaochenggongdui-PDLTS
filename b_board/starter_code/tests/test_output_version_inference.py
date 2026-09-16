"""公开输出阶段推断一致性回归测试。

训练和推理落点由 ``src/system/pdlts_light.py`` 决定，评测和打包落点由
``scripts/shared/output_layout.py`` 决定。两份实现分立，但必须对公开阶段保持一致。
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_STARTER)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)
_SCRIPTS_SHARED = os.path.join(_ROOT, "scripts", "shared")
if _SCRIPTS_SHARED not in sys.path:
    sys.path.insert(0, _SCRIPTS_SHARED)

from src.system.pdlts_light import (
    _infer_output_version as train_infer,
    _versioned_output_dir,
)
from output_layout import infer_output_version as eval_infer


PUBLIC_CASES = [
    ("20260904_b_final_pass1_base_official_b_predict", "b_final"),
    ("20260904_b_final_pass2_specialist_cascade_official_b_predict", "b_final"),
    ("20260904_a_final_reproduce_submission", "a_final"),
]


def test_public_stage_maps_on_both_sides():
    """公开阶段标签直接归入公开输出层。"""
    for name, expected in PUBLIC_CASES:
        assert train_infer(name) == expected
        assert eval_infer(name) == expected


def test_unversioned_is_rejected():
    """无公开阶段标记时返回未分层状态。"""
    assert train_infer("run_without_public_stage_token") == "_unversioned"
    assert eval_infer("run_without_public_stage_token") == "_unversioned"


def test_managed_output_root_requires_public_stage_or_config():
    """受管输出根不能静默创建未分层目录。"""
    out = _versioned_output_dir(
        "../outputs/runs",
        "run_without_public_stage_token",
        config_version="b_final",
    ).replace("\\", "/")
    assert out.endswith("../outputs/runs/b_final/run_without_public_stage_token")
    try:
        _versioned_output_dir("../outputs/runs", "run_without_public_stage_token")
    except ValueError as exc:
        assert "cannot infer output version" in str(exc)
    else:
        raise AssertionError("managed outputs root accepted unversioned run_id")


def test_train_eval_implementations_agree():
    """两份分立实现对公开输入域逐例一致。"""
    for name, _ in PUBLIC_CASES:
        assert train_infer(name) == eval_infer(name)


if __name__ == "__main__":
    test_public_stage_maps_on_both_sides()
    test_unversioned_is_rejected()
    test_managed_output_root_requires_public_stage_or_config()
    test_train_eval_implementations_agree()
    print("ALL PUBLIC OUTPUT STAGE TESTS PASSED")
