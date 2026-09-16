"""统一 outputs 公开阶段分层规则。"""

from __future__ import annotations

import re
from pathlib import Path


def infer_output_version(name: str) -> str:
    """从 run_id/artifact_id 推断公开榜单阶段。"""
    for public_version in ("a_final", "b_final"):
        if re.search(r"(^|_)" + re.escape(public_version) + r"($|_)", name):
            return public_version
    return "_unversioned"


def versioned_artifact_dir(root: Path | str, artifact_id: str) -> Path:
    """返回 outputs/<kind>/<stage>/<artifact_id>。"""
    root = Path(root)
    version = infer_output_version(artifact_id)
    if version == "_unversioned":
        raise ValueError(
            "cannot infer output version for artifact_id; include a public stage "
            "such as b_final or a_final"
        )
    return root / version / artifact_id


def latest_artifact(root: Path | str, pattern: str) -> Path | None:
    """在版本化布局中查找最新 artifact；兼容旧平铺路径作为兜底。"""
    root = Path(root)
    matches = list(root.glob(f"*/*{pattern}*")) + list(root.glob(f"*{pattern}*"))
    matches = [p for p in matches if p.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)
