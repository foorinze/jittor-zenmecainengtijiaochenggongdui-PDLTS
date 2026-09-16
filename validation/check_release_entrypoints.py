"""在隔离目录验证 A/B 入口帮助与缺失数据报错，不运行模型。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bash", default="bash")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    sources = {"a": "a_board/run_all.sh", "b": "b_board/reproduce_b_final.sh"}
    with tempfile.TemporaryDirectory(prefix="pdlts_release_entry_") as temporary:
        base = Path(temporary)
        for board, relative in sources.items():
            target = base / (board + "_board") / Path(relative).name
            target.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / relative, target)
        b_root = base / "b_board"
        (b_root / "checkpoints").mkdir()
        for name in ("base_ep149.pkl", "specialist_final.pkl"):
            (b_root / "checkpoints" / name).write_text("preflight fixture, not a model", encoding="utf-8")
        (b_root / "starter_code").mkdir()
        (b_root / "starter_code/run.py").write_text("raise RuntimeError('model must not run')\n", encoding="utf-8")
        cases = [
            ("b", ["--help"], 0, "--base-ckpt"),
            ("b", ["--unknown"], 2, "unknown argument"),
            ("b", [], 1, "B 榜测试数据未找到"),
            ("a", [], 2, "full|infer"),
            ("a", ["infer"], 1, "dataset/test_noisy"),
            ("a", ["full"], 1, "dataset/test_noisy"),
        ]
        for board, options, expected, marker in cases:
            script = base / (board + "_board") / Path(sources[board]).name
            result = subprocess.run([args.bash, script.as_posix(), *options], cwd=base,
                                    capture_output=True, text=True, encoding="utf-8", timeout=30)
            passed = result.returncode == expected and marker in result.stdout + result.stderr
            records.append({"board": board, "arguments": options, "exit_code": result.returncode,
                            "expected_exit_code": expected, "expected_message": marker, "passed": passed})
        a_data = base / "a_board/dataset/test_noisy"
        a_data.mkdir(parents=True)
        result = subprocess.run([args.bash, (base / "a_board/run_all.sh").as_posix(), "full"],
                                cwd=base, capture_output=True, text=True, encoding="utf-8", timeout=30)
        records.append({"board": "a", "arguments": ["full"], "fixture": "test data directory only",
                        "exit_code": result.returncode,
                        "passed": result.returncode == 1 and "dataset/train" in result.stderr})
    data = {"schema_version": 1, "scope": "隔离目录入口检查，未运行训练或推理",
            "sources": [{"path": p, "sha256": hashlib.sha256((ROOT / p).read_bytes()).hexdigest()} for p in sources.values()],
            "cases": records, "passed": all(r["passed"] for r in records)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(data, ensure_ascii=False))
    return 0 if data["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
