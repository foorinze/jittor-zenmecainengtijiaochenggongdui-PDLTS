#!/usr/bin/env python3
"""A/B 榜通用 MOCK 评测驱动。

这个脚本只负责三件事：生成 MOCK 样本清单、调用对应阶段的
``evaluate_mock.py``、把命令和结果归档到 ``outputs/evals/<stage>/``。
它不修改模型输出，也不把 MOCK 分数当作线上成绩。

路径规则：相对路径均相对于 ``b_board/`` 目录解析，因此从仓库根目录或
``b_board/starter_code/`` 运行都不会把产物写到错误位置。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


SCOPE_TO_DATALIST = {
    "mock20": "datalist/mock.txt",
    "mock200": "datalist/mock_full.txt",
    "mock_b": "datalist/mock_b.txt",
}
SCOPE_TO_EXPECTED_COUNT = {"mock20": 20, "mock200": 200}


def _find_project_root() -> Path:
    """找到同时包含 starter_code 和 scripts 的公开 code 根目录。"""
    cur = Path(__file__).resolve().parent
    while True:
        if (cur / "starter_code").is_dir() and (cur / "scripts").is_dir():
            return cur
        if cur.parent == cur:
            raise RuntimeError("cannot locate public code root")
        cur = cur.parent


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _infer_stage(value: str) -> str:
    for stage in ("a_final", "b_final"):
        if stage in value:
            return stage
    return ""


def _infer_scope_from_name(value: str) -> str:
    name = Path(value).name.lower()
    if "mock_b" in name:
        return "mock_b"
    if "mock_full" in name or "mock200" in name:
        return "mock200"
    if name == "mock.txt" or "mock20" in name:
        return "mock20"
    return ""


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read JSON manifest: {path}: {exc}") from exc


def _build_datalist(args: argparse.Namespace, root: Path) -> int:
    mock_dir = _resolve_path(args.mock_dir, root)
    shapenet_root = mock_dir / "shapenet"
    if not shapenet_root.is_dir():
        print(f"[FAIL] mock/shapenet not found: {shapenet_root}")
        return 2

    entries: list[str] = []
    for class_dir in sorted(p for p in shapenet_root.iterdir() if p.is_dir()):
        for sample_dir in sorted(p for p in class_dir.iterdir() if p.is_dir()):
            if (sample_dir / "noisy.npy").is_file() and (sample_dir / "clean.npy").is_file():
                rel = sample_dir.relative_to(mock_dir).as_posix()
                entries.append(rel)

    if args.limit > 0:
        entries = entries[: args.limit]

    starter_root = _resolve_path(args.starter_root, root)
    if args.out:
        out_path = _resolve_path(args.out, root)
    else:
        filename = "mock.txt" if args.limit > 0 else "mock_full.txt"
        out_path = starter_root / "datalist" / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(f"{entry}\n" for entry in entries), encoding="utf-8")
    print(f"[OK] wrote {len(entries)} entries to {out_path}")
    return 0


def _parse_metrics(stdout: str) -> dict:
    metrics: dict[str, int | float] = {}

    def parse_number(line: str, as_int: bool = False):
        try:
            value = line.split(":")[-1].split("/")[0].strip()
            return int(value) if as_int else float(value)
        except (ValueError, IndexError):
            return None

    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("评测样本总数:"):
            value = parse_number(line, as_int=True)
            if value is not None:
                metrics["total_samples"] = value
        elif line.startswith("有效预测数:"):
            value = parse_number(line, as_int=True)
            if value is not None:
                metrics["valid_predictions"] = value
        elif line.startswith("缺失预测数:"):
            value = parse_number(line, as_int=True)
            if value is not None:
                metrics["missing_predictions"] = value
        elif line.startswith("点数不匹配数:"):
            value = parse_number(line, as_int=True)
            if value is not None:
                metrics["shape_errors"] = value
        elif line.startswith("缺失 mesh/norm 数:"):
            value = parse_number(line, as_int=True)
            if value is not None:
                metrics["skipped_mesh_norm"] = value
        elif line.startswith("平均 CD_pred:"):
            value = parse_number(line)
            if value is not None:
                metrics["cd_pred_mean"] = value
        elif line.startswith("平均 CD_noisy:"):
            value = parse_number(line)
            if value is not None:
                metrics["cd_noisy_mean"] = value
        elif line.startswith("平均 P2S_pred:"):
            value = parse_number(line)
            if value is not None:
                metrics["p2s_pred_mean"] = value
        elif line.startswith("平均 P2S_noisy:"):
            value = parse_number(line)
            if value is not None:
                metrics["p2s_noisy_mean"] = value
        elif line.startswith("CD 得分:"):
            value = parse_number(line)
            if value is not None:
                metrics["cd_score"] = value
        elif line.startswith("P2S 得分:"):
            value = parse_number(line)
            if value is not None:
                metrics["p2s_score"] = value
        elif line.startswith("最终得分"):
            value = parse_number(line)
            if value is not None:
                metrics["final_score"] = value
    return metrics


def _check_scope_counts(metrics: dict, scope: str) -> tuple[bool, str]:
    expected = SCOPE_TO_EXPECTED_COUNT.get(scope)
    if expected is None:
        return True, ""
    actual = metrics.get("total_samples")
    valid = metrics.get("valid_predictions")
    missing = metrics.get("missing_predictions")
    if actual != expected or valid != expected or missing != 0:
        return False, (
            f"{scope} requires total={expected}, valid={expected}, missing=0; "
            f"got total={actual}, valid={valid}, missing={missing}"
        )
    return True, ""


def _run_eval(args: argparse.Namespace, root: Path) -> int:
    predict_run = _resolve_path(args.predict_run, root)
    if not predict_run.is_dir():
        print(f"[FAIL] predict-run is not a directory: {predict_run}")
        return 2

    is_golden = args.pred_filename != "denoised.npy"
    if is_golden:
        pred_root = predict_run
        verdict = "golden"
        predict_manifest = {}
    else:
        pred_root = predict_run / "pred"
        if not pred_root.is_dir():
            print(f"[FAIL] missing prediction directory: {pred_root}")
            return 2
        predict_manifest = _read_json(predict_run / "manifest.json")
        verdict = (predict_manifest.get("summary") or {}).get("verdict", "unknown")
        if verdict in {"red", "unknown"} and not args.force:
            print(f"[REFUSE] predict verdict={verdict}; use --force only for debugging")
            return 3
        if verdict == "yellow":
            print("[WARN] predict verdict=yellow; continue with explicit warning")

    stage = args.stage or _infer_stage(str(predict_run)) or "b_final"
    starter_root = _resolve_path(args.starter_root, root)
    evaluator = starter_root / "evaluate_mock.py"
    if not evaluator.is_file():
        print(f"[FAIL] evaluate_mock.py not found: {evaluator}")
        return 2

    mock_dir = _resolve_path(args.mock_dir, root)
    mesh_dir = _resolve_path(args.mesh_dir, root) if args.mesh_dir else None
    if mesh_dir is None:
        print("[FAIL] --mesh-dir is required: precise P2S needs mesh + norm.json")
        return 2

    scope = args.scope or ""
    count_scope = args.scope or ""
    datalist = _resolve_path(args.datalist, root) if args.datalist else None
    if datalist is not None:
        inferred = _infer_scope_from_name(str(datalist))
        if inferred and scope and inferred != scope:
            print(f"[FAIL] scope={scope} conflicts with datalist={inferred}")
            return 2
        scope = scope or inferred
        count_scope = count_scope or inferred
    if not scope and not is_golden:
        manifest_scope = str(predict_manifest.get("scope") or "").strip()
        if manifest_scope in {"mock20", "mock200", "mock_b"}:
            scope = manifest_scope
    if datalist is None and scope in SCOPE_TO_DATALIST:
        candidate = starter_root / SCOPE_TO_DATALIST[scope]
        if candidate.is_file():
            datalist = candidate

    eval_id = args.eval_id
    if not eval_id:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        eval_id = f"{stamp}_{stage}_{scope or 'mock'}_eval"
    eval_dir = _resolve_path(args.eval_root, root) / stage / eval_id
    (eval_dir / "logs").mkdir(parents=True, exist_ok=True)

    eval_cmd = [
        sys.executable,
        str(evaluator),
        "--pred_dir", str(pred_root),
        "--gt_dir", str(mock_dir),
        "--noisy_dir", str(mock_dir),
        "--mesh_dir", str(mesh_dir),
        "--pred_filename", args.pred_filename,
        "--gt_filename", "clean.npy",
        "--noisy_filename", "noisy.npy",
        "--norm_filename", "norm.json",
    ]
    if datalist is not None:
        eval_cmd += ["--datalist", str(datalist)]
    if args.workers is not None:
        eval_cmd += ["--workers", str(args.workers)]
    if args.verbose:
        eval_cmd.append("--verbose")
    if args.allow_missing_mesh_norm:
        eval_cmd.append("--allow-missing-mesh-norm")

    (eval_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\n" + shlex.join(eval_cmd) + "\n", encoding="utf-8"
    )
    print("[run] " + shlex.join(eval_cmd))
    result = subprocess.run(
        eval_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(starter_root),
    )
    (eval_dir / "logs" / "eval.log").write_text(result.stdout, encoding="utf-8")
    (eval_dir / "report.txt").write_text(result.stdout, encoding="utf-8")
    metrics = _parse_metrics(result.stdout)

    # 只有显式 scope/datalist 才是业务验收；预测 manifest 中的 scope 只用于记录，
    # 这样小型合成单测可以使用 3 个样本而不会伪装成 mock20 完整验收。
    count_ok, count_message = _check_scope_counts(metrics, count_scope)
    if not count_ok:
        print(f"[FAIL] {count_message}")

    manifest = {
        "eval_id": eval_id,
        "kind": "eval",
        "stage": stage,
        "scope": scope,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predict_run": str(predict_run),
        "predict_verdict": verdict,
        "predict_manifest_summary": predict_manifest.get("summary", {}),
        "starter_root": str(starter_root),
        "mock_dir": str(mock_dir),
        "mesh_dir": str(mesh_dir),
        "datalist": str(datalist) if datalist is not None else "",
        "pred_filename": args.pred_filename,
        "is_golden_mode": is_golden,
        "eval_returncode": result.returncode,
        "scope_count_check": {
            "scope": count_scope,
            "ok": count_ok,
            "message": count_message,
        },
        "metrics": metrics,
    }
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (eval_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if result.returncode != 0:
        print(f"[FAIL] evaluate_mock.py returned {result.returncode}")
        return result.returncode
    if not count_ok:
        return 2
    print(f"[OK] eval artifacts written to {eval_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B 榜公开 MOCK 评测驱动")
    parser.add_argument("--build-datalist", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--predict-run", default="")
    parser.add_argument("--stage", choices=("a_final", "b_final"), default="")
    parser.add_argument("--starter-root", default="starter_code")
    parser.add_argument("--mock-dir", default="dataset/mock_test")
    parser.add_argument("--mesh-dir", default="dataset/train")
    parser.add_argument("--datalist", default="")
    parser.add_argument("--scope", choices=("mock20", "mock200", "mock_b"), default="")
    parser.add_argument("--eval-id", default="")
    parser.add_argument("--eval-root", default="outputs/evals")
    parser.add_argument("--pred-filename", default="denoised.npy")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--allow-missing-mesh-norm", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = _find_project_root()
    if args.build_datalist:
        return _build_datalist(args, root)
    if not args.predict_run:
        print("[FAIL] --predict-run is required unless --build-datalist is used")
        return 2
    return _run_eval(args, root)


if __name__ == "__main__":
    raise SystemExit(main())
