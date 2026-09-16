"""使用合成点云检查 A/B 评测器，结果不代表竞赛成绩。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_fixture(root, case):
    keys = ["shapenet/00000000/sample_0", "shapenet/00000000/sample_1"]
    clean = np.array([
        [-0.6, -0.4, 0], [0.4, -0.4, 0], [0.4, 0.6, 0],
        [-0.6, 0.6, 0], [-0.2, 0.1, 0], [0.2, 0.3, 0],
    ], dtype=np.float64)
    noisy = clean + np.array([0.015, -0.01, 0.12])
    center = np.array([3.0, -4.0, 2.0])
    scale = 2.5
    mesh_vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    mesh_vertices = mesh_vertices * scale + center
    for index, key in enumerate(keys):
        sample = root / "samples" / key
        sample.mkdir(parents=True)
        if case != "missing_clean" or index == 0:
            np.save(sample / "clean.npy", clean)
        if case != "missing_noisy" or index == 0:
            np.save(sample / "noisy.npy", noisy)
        if case != "missing_norm" or index == 0:
            (sample / "norm.json").write_text(
                json.dumps({"center": center.tolist(), "scale": scale}), encoding="utf-8"
            )
        if case != "missing_mesh" or index == 0:
            mesh_dir = root / "meshes" / key / "models"
            mesh_dir.mkdir(parents=True)
            vertices = "".join("v " + " ".join(str(v) for v in p) + "\n" for p in mesh_vertices)
            (mesh_dir / "model_normalized.obj").write_text(
                vertices + "f 1 2 3\nf 1 3 4\n", encoding="utf-8"
            )
        if case == "missing_prediction" and index == 1:
            continue
        pred = noisy.copy() if case == "noisy" else clean.copy()
        if case == "shape_error" and index == 1:
            pred = pred[:-1]
        if case == "nonfinite" and index == 1:
            pred[0, 0] = np.nan
        destination = root / "predictions" / key
        destination.mkdir(parents=True)
        np.save(destination / "denoised.npy", pred)
    (root / "datalist.txt").write_text("\n".join(keys) + "\n", encoding="utf-8")


def run_case(evaluator, case, parse_metrics):
    with tempfile.TemporaryDirectory(prefix="pdlts_release_eval_") as directory:
        fixture = Path(directory)
        base_case = case.removesuffix("_scan")
        make_fixture(fixture, base_case)
        command = [
            sys.executable, "-B", str(evaluator),
            "--pred_dir", str(fixture / "predictions"),
            "--gt_dir", str(fixture / "samples"),
            "--noisy_dir", str(fixture / "samples"),
            "--mesh_dir", str(fixture / "meshes"),
            "--datalist", str(fixture / "datalist.txt"), "--workers", "1",
        ]
        if case.endswith("_scan"):
            position = command.index("--datalist")
            del command[position:position + 2]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=env, timeout=60)
        metrics = parse_metrics(result.stdout)
        rejected = base_case in {"missing_mesh", "missing_norm", "missing_clean", "missing_noisy"}
        if rejected:
            passed = result.returncode != 0 and "[FAIL]" in result.stdout + result.stderr
        else:
            target = 0.0 if case == "noisy" else 50.0 if case in {
                "missing_prediction", "shape_error", "nonfinite"
            } else 100.0
            passed = (
                result.returncode == 0
                and metrics.get("total_samples") == 2
                and abs(metrics.get("final_score", -1000) - target) < 0.01
            )
            if case == "missing_prediction":
                passed = passed and metrics.get("missing_predictions") == 1
            if case in {"shape_error", "nonfinite"}:
                passed = passed and metrics.get("shape_errors") == 1
        return {"case": case, "passed": bool(passed), "exit_code": result.returncode, "metrics": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    driver = load_module("release_mock_driver", ROOT / "b_board/scripts/shared/run_mock_eval.py")
    cases = ["clean", "noisy", "missing_prediction", "shape_error", "nonfinite",
             "missing_mesh", "missing_norm", "missing_clean", "missing_noisy",
             "missing_clean_scan", "missing_noisy_scan"]
    boards = []
    for board, relative in [
        ("a_final", "a_board/starter_code/evaluate_mock.py"),
        ("b_final", "b_board/starter_code/evaluate_mock.py"),
    ]:
        evaluator = ROOT / relative
        module = load_module("release_eval_" + board, evaluator)
        records = [run_case(evaluator, case, driver._parse_metrics) for case in cases]
        boards.append({"board": board, "evaluator": relative,
                       "sha256": hashlib.sha256(evaluator.read_bytes()).hexdigest(),
                       "p2s_backend": "pcu-BVH" if module.HAS_PCU else "numpy-triangle-fallback",
                       "cases": records})
        print(board, " ".join(r["case"] + ":" + ("PASS" if r["passed"] else "FAIL") for r in records))
    result = {"schema_version": 1, "scope": "合成数据上的评测契约验证，不是模型评分或线上成绩",
              "python": platform.python_version(), "system": platform.system(), "boards": boards,
              "passed": all(r["passed"] for b in boards for r in b["cases"])}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
