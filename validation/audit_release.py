"""检查待发布文件、语法、引用和权重；不执行训练或网络发布。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml
from release_inventory import EXCLUDED_FILES, MANIFEST_PATH, release_files


ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv"}
TEXT_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".json", ".txt", ".cff"}
PATTERNS = {
    "private_key": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"),
    "access_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-proj-[A-Za-z0-9_-]{30,})"),
    "internal_version": re.compile(r"\bpdlts_\d+_\d+\b"),
    "machine_path": re.compile(r"(?:[A-Z]:[\\/](?:Users|Projects|Documents)[\\/]|/home/[A-Za-z0-9_.-]+/|/data/[A-Z][A-Z0-9_-]*/|/mnt/[a-z]/(?:Projects|Documents)/)"),
    "phone_number": re.compile(r"(?<![A-Za-z0-9_.])1[3-9]\d{9}(?![A-Za-z0-9_.])"),
    "email_address": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bash", default="bash")
    args = parser.parse_args()
    findings = []
    audited_files = []
    counts = {"files": 0, "python": 0, "json": 0, "yaml": 0, "shell": 0, "links": 0}
    files = list(release_files(ROOT))
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if path.resolve() == args.output.resolve() or relative == MANIFEST_PATH:
            continue
        audited_files.append({"path": relative, "sha256": sha256(path)})
        counts["files"] += 1
        if path.suffix in {".pyc", ".pyo"} or path.stat().st_size == 0:
            # Empty package markers are source files, not generated artifacts.
            if path.name != "__init__.py":
                findings.append({"path": relative, "kind": "unwanted_artifact"})
        if path.suffix not in TEXT_SUFFIXES and path.name not in {"LICENSE", ".gitignore"}:
            continue
        text = path.read_text(encoding="utf-8-sig")
        for number, line in enumerate(text.splitlines(), 1):
            for kind, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append({"path": relative, "line": number, "kind": kind})
        try:
            if path.suffix == ".py":
                ast.parse(text, filename=relative)
                counts["python"] += 1
            elif path.suffix == ".json":
                json.loads(text)
                counts["json"] += 1
            elif path.suffix in {".yaml", ".yml", ".cff"}:
                yaml.safe_load(text)
                counts["yaml"] += 1
            elif path.suffix == ".sh":
                checked = subprocess.run([args.bash, "-n", str(path)], capture_output=True)
                if checked.returncode:
                    findings.append({"path": relative, "kind": "shell_syntax"})
                counts["shell"] += 1
        except (SyntaxError, ValueError, yaml.YAMLError):
            findings.append({"path": relative, "kind": "syntax"})
        if path.suffix == ".md":
            # Only ordinary Markdown links are treated as file references.
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
                target = target.split("#", 1)[0]
                if not target or re.match(r"[a-z]+://|mailto:", target):
                    continue
                if not (path.parent / target).exists():
                    findings.append({"path": relative, "kind": "broken_link", "target": target})
                counts["links"] += 1
    weights = []
    manifest = json.loads((ROOT / "b_board/checkpoints/checkpoint_manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["checkpoints"].values():
        path = ROOT / entry["file"]
        weights.append({"path": entry["file"], "sha256": sha256(path),
                        "passed": sha256(path) == entry["sha256"] and path.stat().st_size == entry["bytes"]})
    a_weights = ROOT / "a_board/checkpoints"
    for line in (a_weights / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        path = a_weights / filename.lstrip("*")
        weights.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                        "passed": sha256(path) == expected})
    result = {"schema_version": 1, "scope": "发布副本静态检查与权重校验",
              "counts": counts, "findings": findings, "weights": weights, "audited_files": audited_files,
              "excluded_files": sorted(EXCLUDED_FILES),
              "passed": not findings and all(w["passed"] for w in weights)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "findings": findings, "weights_passed": all(w["passed"] for w in weights)}, ensure_ascii=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
