"""验证 A/B 模型在无法导入已排除 DCD 文件时的损失与梯度。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BOARDS = {
    "a_final": "a_board/starter_code",
    "b_final": "b_board/starter_code",
}


def check_board(board):
    class BlockDcd(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.endswith(".dcd_official"):
                raise ImportError("DCD source is intentionally unavailable")
            return None

    sys.meta_path.insert(0, BlockDcd())
    sys.path.insert(0, str(ROOT / BOARDS[board]))
    import jittor as jt
    import numpy as np
    from src.model.pdlts_light.system import PDLTSLight
    from src.model.pdlts_light import losses

    jt.flags.use_cuda = 0
    jt.set_global_seed(42)
    assert losses.__all__ == ["compute_dcd_like_loss"]
    cfg = {"num_neighbors": 8, "mlgc_hidden": 32, "coupling_hidden": 32}
    model = PDLTSLight(cfg, {})
    clean = jt.randn(2, 1, 32, 3)
    batch = {"pc_clean": clean, "pc_noisy": clean + 0.01 * jt.randn(2, 1, 32, 3)}
    loss = model.training_step(batch)
    assert "chamfer" in loss and "l2" in loss and "dcd_official" not in loss
    values = {name: float(value.item()) for name, value in loss.items()}
    assert all(np.isfinite(value) for value in values.values())
    parameters = [
        (name, value) for name, value in model.network.named_parameters()
        if name.endswith(".weight")
    ]
    gradients = jt.grad(loss["chamfer"] + 0.1 * loss["l2"], [value for _, value in parameters])
    nonzero = []
    for (name, _), gradient in zip(parameters, gradients):
        array = gradient.numpy()
        assert np.isfinite(array).all(), name
        if np.any(array != 0):
            nonzero.append(name)
    assert nonzero, "No weight receives a nonzero gradient"
    try:
        PDLTSLight(dict(cfg, dcd_official_loss_mode="on"), {})
    except ValueError as error:
        assert "excluded from this release" in str(error)
    else:
        raise AssertionError("Excluded DCD configuration was accepted")
    assert not any(name.endswith(".dcd_official") for name in sys.modules)
    return {"board": board, "losses": values, "weights_checked": len(parameters),
            "weights_with_nonzero_gradient": len(nonzero),
            "finite_nonzero_gradient": True,
            "excluded_mode_rejected": True, "dcd_module_not_imported": True, "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--board", choices=BOARDS)
    args = parser.parse_args()
    if args.board:
        print("RELEASE_MODEL_RESULT=" + json.dumps(check_board(args.board)))
        return 0
    if args.output is None:
        parser.error("--output is required")
    results = []
    for board in BOARDS:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", DISABLE_MULTIPROCESSING="1")
        proc = subprocess.run(
            [sys.executable, "-B", str(Path(__file__).resolve()), "--board", board],
            env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        lines = [line for line in proc.stdout.splitlines() if line.startswith("RELEASE_MODEL_RESULT=")]
        if proc.returncode or len(lines) != 1:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            return 1
        results.append(json.loads(lines[0].split("=", 1)[1]))
    paths = [Path(__file__).resolve()]
    for base in BOARDS.values():
        paths.extend([ROOT / base / "src/model/pdlts_light/system.py",
                      ROOT / base / "src/model/pdlts_light/losses/__init__.py"])
    report = {
        "schema_version": 1, "scope": "Jittor CPU 合成小点云；隔离 DCD 导入，检查 A/B 默认损失与梯度及禁用配置",
        "python": platform.python_version(), "system": platform.system(),
        "sources": [{"path": p.relative_to(ROOT).as_posix(),
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        "boards": results, "passed": all(item["passed"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"boards": results, "passed": report["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
