"""outputs 目录的公开阶段分层规则。

所有运行产物按 outputs/<kind>/<stage>/<artifact_id>/ 归档，stage 从
artifact_id 里的公开阶段名解析。这样一份产物的归属只有一个来源，不依赖
目录是哪一步建的。
"""

from __future__ import annotations

import re
from pathlib import Path


def infer_output_version(name: str) -> str:
    """从 run_id / artifact_id 解析公开榜单阶段。"""
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
            "cannot infer output version for artifact_id; "
            "include a public stage such as a_final or b_final"
        )
    return root / version / artifact_id


def latest_artifact(root: Path | str, pattern: str) -> Path | None:
    """在公开阶段目录中查找最新 artifact。"""
    root = Path(root)
    matches = list(root.glob(f"*/*{pattern}*")) + list(root.glob(f"*{pattern}*"))
    matches = [p for p in matches if p.is_dir()]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)
