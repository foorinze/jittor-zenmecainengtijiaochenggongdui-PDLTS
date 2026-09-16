"""mock 评测流水线单测。

运行:
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_pdlts_light_mock_eval

验证内容:
    A. --build-datalist
        A1. 扫合成 mock_test 目录, 正确写出 datalist/mock.txt
        A2. --limit N 只取前 N 个
        A3. 只保留同时有 noisy.npy + clean.npy 的 sample
    B. run eval
        B1. verdict=green: 调子进程 evaluate_mock.py，产物 manifest/metrics/report/command
        B2. verdict=red: 默认 refuse
        B3. verdict=red + --force: 放行
        B4. metrics 能从模拟 evaluate.py stdout 中正确解析
        B5. eval_dir 里 command.sh / manifest.json / report.txt / logs/eval.log 齐全

单测模拟真实 evaluate_mock.py：通过创建临时 fake evaluator 脚本
并用 monkey patch run_mock_eval._find_project_root 指向临时 root.
"""

import json
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np


# evaluate_mock.py 打印的格式必须和 run_mock_eval._parse_metrics 匹配
FAKE_EVALUATE_SCRIPT = textwrap.dedent('''
    #!/usr/bin/env python3
    """Fake evaluate_mock.py for test_pdlts_light_mock_eval. 打印 run_mock_eval 期待的格式."""
    import argparse
    import sys

    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--noisy_dir", required=True)
    p.add_argument("--mesh_dir", default="")
    p.add_argument("--pred_filename", default="denoised.npy")
    p.add_argument("--gt_filename", default="clean.npy")
    p.add_argument("--noisy_filename", default="noisy.npy")
    p.add_argument("--norm_filename", default="norm.json")
    p.add_argument("--mesh_data_name", default="models/model_normalized.obj")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # 打印一段假报告, 数字便于单测 assert
    print("加速后端: CD=scipy, P2S=fake")
    print("评测样本总数:  3")
    print("评测耗时: 0.1s")
    print("平均 CD_pred:      0.00123")
    print("平均 CD_noisy:     0.00567")
    print("CD 得分:            42.50 / 100.00")
    if args.mesh_dir:
        print("平均 P2S_pred:     0.00234")
        print("平均 P2S_noisy:    0.00890")
        print("P2S 得分:           35.20 / 100.00")
        print("最终得分 (0.5×CD + 0.5×P2S):  38.85 / 100.00")
    else:
        print("最终得分 (CD):      42.50 / 100.00")
    sys.exit(0)
''').lstrip()


def _make_synthetic_pdlts_root() -> str:
    """造一个迷你 PDLTS root:
        <root>/scripts/shared/run_mock_eval.py  (copy 真实脚本)
        <root>/starter_code/evaluate_mock.py  (fake)
        <root>/starter_code/datalist/
        <root>/dataset/mock_test/shapenet/...
    """
    tmp = tempfile.mkdtemp(prefix="mock_eval_eval_root_")
    starter = os.path.join(tmp, "starter_code")
    os.makedirs(os.path.join(starter, "datalist"), exist_ok=True)
    # Fake evaluator 文件名对齐 run_mock_eval 期望的路径 (evaluate_mock.py).
    with open(os.path.join(starter, "evaluate_mock.py"), "w", encoding="utf-8") as f:
        f.write(FAKE_EVALUATE_SCRIPT)
    scripts = os.path.join(tmp, "scripts", "shared")
    os.makedirs(scripts, exist_ok=True)
    real_script = os.path.join(_STARTER, "..", "scripts", "shared", "run_mock_eval.py")
    shutil.copy(real_script, os.path.join(scripts, "run_mock_eval.py"))
    real_layout = os.path.join(_STARTER, "..", "scripts", "shared", "output_layout.py")
    shutil.copy(real_layout, os.path.join(scripts, "output_layout.py"))
    return tmp


def _make_mock_data_dir(root: str, n_samples: int = 3) -> str:
    """在 root/dataset/mock_test 下造 n_samples 个合成样本."""
    mock = os.path.join(root, "dataset", "mock_test")
    os.makedirs(os.path.join(mock, "shapenet"), exist_ok=True)
    for i in range(n_samples):
        sample = os.path.join(
            mock, "shapenet", "0440", f"sample_{i}",
        )
        os.makedirs(sample, exist_ok=True)
        noisy = np.random.randn(100, 3).astype(np.float32)
        clean = noisy + np.random.randn(100, 3).astype(np.float32) * 0.01
        np.save(os.path.join(sample, "noisy.npy"), noisy)
        np.save(os.path.join(sample, "clean.npy"), clean)
    # 额外造一个只有 noisy 没有 clean 的, 应该被 --build-datalist 过滤
    bad = os.path.join(mock, "shapenet", "0440", "bad_no_clean")
    os.makedirs(bad, exist_ok=True)
    np.save(os.path.join(bad, "noisy.npy"),
            np.random.randn(100, 3).astype(np.float32))
    return mock


def _make_mock_predict_run(root: str, verdict: str = "green", n_samples: int = 3) -> str:
    """造一个假的 predict_run 目录, 包含 pred/shapenet/<cls>/<id>/denoised.npy."""
    run_id = f"pr_{verdict}_mock20_predict"
    predict_run = os.path.join(root, "outputs", "predictions", "b_final", run_id)
    pred_root = os.path.join(predict_run, "pred", "shapenet", "0440")
    os.makedirs(pred_root, exist_ok=True)
    for i in range(n_samples):
        sdir = os.path.join(pred_root, f"sample_{i}")
        os.makedirs(sdir, exist_ok=True)
        np.save(os.path.join(sdir, "denoised.npy"),
                np.random.randn(100, 3).astype(np.float32))
    mani = {
        "run_id": run_id,
        "kind": "predict",
        "scope": "mock20",
        "summary": {"verdict": verdict, "n_samples": n_samples},
    }
    with open(os.path.join(predict_run, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(mani, f, indent=2)
    return predict_run


def _run_mock_eval(root: str, extra: list) -> subprocess.CompletedProcess:
    script = os.path.join(root, "scripts", "shared", "run_mock_eval.py")
    cmd = [sys.executable, script] + extra
    return subprocess.run(cmd, capture_output=True, text=True)


# ============================================================================
# A. build-datalist
# ============================================================================

def test_build_datalist_full():
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        r = _run_mock_eval(root, [
            "--build-datalist",
            "--mock-dir", "dataset/mock_test",
        ])
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
        mock_txt = os.path.join(root, "starter_code", "datalist", "mock_full.txt")
        assert os.path.exists(mock_txt)
        with open(mock_txt, encoding="utf-8") as f:
            entries = [l.strip() for l in f if l.strip()]
        # 应该有 3 条 (bad_no_clean 被过滤)
        assert len(entries) == 3, f"expected 3 entries, got {entries}"
        for e in entries:
            assert e.startswith("shapenet/0440/sample_"), e
            assert "bad_no_clean" not in e
        print("[PASS] test_build_datalist_full")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_build_datalist_limit():
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=5)
        r = _run_mock_eval(root, [
            "--build-datalist",
            "--mock-dir", "dataset/mock_test",
            "--limit", "2",
        ])
        assert r.returncode == 0, r.stderr
        mock_txt = os.path.join(root, "starter_code", "datalist", "mock.txt")
        with open(mock_txt, encoding="utf-8") as f:
            entries = [l.strip() for l in f if l.strip()]
        assert len(entries) == 2, f"expected 2 (limit), got {len(entries)}"
        print("[PASS] test_build_datalist_limit")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ============================================================================
# B. run eval
# ============================================================================

def test_run_eval_green_happy_path():
    """默认路径 (denoised.npy) + 必须有 mesh_dir, evaluate_mock 算 CD+P2S."""
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="green", n_samples=3)

        # 造 mesh_dir 让 evaluate_mock 有合法路径 (fake evaluator 不会真读 mesh,
        # 但 run_mock_eval.py 现在强制 --mesh-dir 非空)
        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        r = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_1",
        ])
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
        eval_dir = os.path.join(root, "outputs", "evals", "b_final", "test_eval_1")
        for fname in ("command.sh", "manifest.json", "metrics.json", "report.txt",
                      "logs/eval.log"):
            assert os.path.exists(os.path.join(eval_dir, fname)), f"missing {fname}"
        with open(os.path.join(eval_dir, "manifest.json"), encoding="utf-8") as f:
            mani = json.load(f)
        assert mani["eval_id"] == "test_eval_1"
        assert mani["kind"] == "eval"
        assert mani["predict_verdict"] == "green"
        assert mani["eval_returncode"] == 0
        assert mani["pred_filename"] == "denoised.npy"
        assert mani["is_golden_mode"] is False
        # Fake evaluator 有 mesh_dir 时走 CD+P2S 路径, 打印最终得分 38.85
        with open(os.path.join(eval_dir, "metrics.json"), encoding="utf-8") as f:
            m = json.load(f)
        assert m.get("cd_score") == 42.5, m
        assert m.get("p2s_score") == 35.2, m
        assert m.get("final_score") == 38.85, m
        print(f"[PASS] test_run_eval_green_happy_path  final={m.get('final_score')}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_eval_refuses_empty_mesh_dir():
    """run_mock_eval 现在强制要 --mesh-dir (evaluate_mock 精确 P2S 的硬要求)."""
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="green", n_samples=3)

        r = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_no_mesh",
        ])
        assert r.returncode != 0, "should refuse empty mesh_dir"
        combined = r.stdout + "\n" + r.stderr
        assert "mesh" in combined.lower()
        print("[PASS] test_run_eval_refuses_empty_mesh_dir")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_eval_golden_mode_clean_as_pred():
    """金标准 drive: --pred-filename=clean.npy + predict_run=mock_test 自身,
    触发 is_golden_mode, 跳过 pred/ 子目录和 verdict 检查.

    这里只验证 run_mock_eval 的金标准驱动逻辑通 (fake evaluator 仍然返回固定数字);
    真正的 100 分验收由独立的真 evaluate_mock 单测做 (test_evaluate_mock_golden_*).
    """
    root = _make_synthetic_pdlts_root()
    try:
        mock_dir = _make_mock_data_dir(root, n_samples=3)
        # 造 mesh_dir
        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        r = _run_mock_eval(root, [
            "--predict-run", mock_dir,    # 指向 mock_test 自身, 不是 predict run
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--pred-filename", "clean.npy",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_golden",
        ])
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
        with open(os.path.join(root, "outputs", "evals", "b_final", "test_eval_golden",
                               "manifest.json"), encoding="utf-8") as f:
            mani = json.load(f)
        assert mani["is_golden_mode"] is True
        assert mani["pred_filename"] == "clean.npy"
        assert mani["predict_verdict"] == "golden"
        print("[PASS] test_run_eval_golden_mode_clean_as_pred")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_eval_refuses_red_verdict():
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="red", n_samples=3)
        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        r = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_red",
        ])
        assert r.returncode != 0, f"should refuse red verdict: {r.stdout}"
        print("[PASS] test_run_eval_refuses_red_verdict")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_eval_force_override_red():
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="red", n_samples=3)
        # 造 mesh_dir (run_mock_eval 强制要)
        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        r = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_forced",
            "--force",
        ])
        assert r.returncode == 0, f"--force should succeed: {r.stderr}"
        eval_dir = os.path.join(root, "outputs", "evals", "b_final", "test_eval_forced")
        assert os.path.exists(os.path.join(eval_dir, "manifest.json"))
        print("[PASS] test_run_eval_force_override_red")
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ============================================================================
# Golden tests: 真跑 evaluate_mock.py (不经 run_mock_eval driver)
# ============================================================================

def _build_golden_fixture():
    """构造一个微型 mock 数据集 + mesh, 用真 evaluate_mock.py 跑 golden 测试.

    关键: clean 点云必须严格在 mesh 表面上 (用 face 重心坐标采样), 否则
    p2s_noisy ≈ p2s_pred (都离表面很远), 打分会失去辨别力.
    """
    tmp = tempfile.mkdtemp(prefix="mock_eval_golden_")
    mock_root = os.path.join(tmp, "dataset", "mock_test", "shapenet", "0440")
    mesh_root = os.path.join(tmp, "dataset", "train", "shapenet", "0440")

    np.random.seed(123)
    for i in range(2):
        sample = f"sample_{i}"
        sdir = os.path.join(mock_root, sample)
        mdir = os.path.join(mesh_root, sample, "models")
        os.makedirs(sdir, exist_ok=True)
        os.makedirs(mdir, exist_ok=True)

        # 一个 cube mesh, 不同中心位置
        verts = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float64) + np.array([i * 10.0, 0, 0])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0],
        ], dtype=np.int32)
        obj_path = os.path.join(mdir, "model_normalized.obj")
        with open(obj_path, "w") as f:
            for v in verts:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for fa in faces:
                f.write(f"f {fa[0]+1} {fa[1]+1} {fa[2]+1}\n")

        # 关键: face 重心坐标采样, 保证点在表面上.
        # clean_world: 原 mesh 坐标系; 然后按 norm.json 变换到单位球空间.
        mesh_center = (verts.max(0) + verts.min(0)) / 2
        mesh_centered = verts - mesh_center
        mesh_scale = np.sqrt((mesh_centered ** 2).sum(1)).max()

        np.random.seed(7 + i)
        clean_world = np.zeros((100, 3), dtype=np.float64)
        for k in range(100):
            face_idx = np.random.randint(0, len(faces))
            u = np.random.rand()
            v = np.random.rand() * (1 - u)
            w = 1 - u - v
            a, b, c = faces[face_idx]
            clean_world[k] = u * verts[a] + v * verts[b] + w * verts[c]

        # 变换到 mock pipeline 的单位球空间
        clean = ((clean_world - mesh_center) / mesh_scale).astype(np.float32)
        np.save(os.path.join(sdir, "clean.npy"), clean)

        # noisy = clean + laplace
        noise = np.random.laplace(0, 0.02, size=clean.shape).astype(np.float32)
        noisy = clean + noise
        np.save(os.path.join(sdir, "noisy.npy"), noisy)

        # norm.json 记真实的 center/scale
        with open(os.path.join(sdir, "norm.json"), "w") as f:
            json.dump({
                "center": mesh_center.tolist(),
                "scale": float(mesh_scale),
            }, f)

    return tmp


def _run_evaluate_mock(tmp, pred_filename, workers=1, evaluator_path=None):
    """直接调 starter_code/evaluate_mock.py (不经 run_mock_eval)."""
    evaluate_mock = evaluator_path or os.path.join(_STARTER, "evaluate_mock.py")
    cmd = [
        sys.executable, evaluate_mock,
        "--pred_dir", os.path.join(tmp, "dataset", "mock_test"),
        "--gt_dir", os.path.join(tmp, "dataset", "mock_test"),
        "--noisy_dir", os.path.join(tmp, "dataset", "mock_test"),
        "--mesh_dir", os.path.join(tmp, "dataset", "train"),
        "--pred_filename", pred_filename,
        "--workers", str(workers),
    ]
    return subprocess.run(cmd, capture_output=True, text=True)


def _extract_final_score(stdout: str) -> float:
    for line in stdout.splitlines():
        if "最终得分" in line and "/" in line:
            try:
                return float(line.split(":")[-1].split("/")[0].strip())
            except (ValueError, IndexError):
                pass
    return -1.0


def test_evaluate_mock_golden_clean_as_pred():
    """金标准 1: clean.npy 作为 pred, 最终分应接近 100."""
    tmp = _build_golden_fixture()
    try:
        r = _run_evaluate_mock(tmp, pred_filename="clean.npy")
        assert r.returncode == 0, f"evaluate_mock failed: {r.stderr}"
        final = _extract_final_score(r.stdout)
        assert final > 95.0, (
            f"clean-as-pred final_score = {final:.2f}, expected > 95.0.\n"
            f"Evaluator coordinate system bug? stdout:\n{r.stdout}"
        )
        print(f"[PASS] test_evaluate_mock_golden_clean_as_pred  final={final:.2f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluate_mock_golden_noisy_as_pred():
    """金标准 2: noisy.npy 作为 pred, 最终分应接近 0 (pred 和 noisy 相同, 没改进)."""
    tmp = _build_golden_fixture()
    try:
        r = _run_evaluate_mock(tmp, pred_filename="noisy.npy")
        assert r.returncode == 0, f"evaluate_mock failed: {r.stderr}"
        final = _extract_final_score(r.stdout)
        # pred == noisy => val_pred/val_noisy == 1 => score = 0
        assert final < 5.0, (
            f"noisy-as-pred final_score = {final:.2f}, expected < 5.0.\n"
            f"Scoring formula bug? stdout:\n{r.stdout}"
        )
        print(f"[PASS] test_evaluate_mock_golden_noisy_as_pred  final={final:.2f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_board_evaluate_mock_golden_contract():
    """A 榜快照必须与 B 榜共享同一 MOCK 坐标和金标准契约."""
    tmp = _build_golden_fixture()
    try:
        a_evaluator = os.path.join(
            _STARTER, "..", "..", "a_board", "starter_code", "evaluate_mock.py"
        )
        clean_run = _run_evaluate_mock(
            tmp, pred_filename="clean.npy", evaluator_path=a_evaluator
        )
        noisy_run = _run_evaluate_mock(
            tmp, pred_filename="noisy.npy", evaluator_path=a_evaluator
        )
        assert clean_run.returncode == 0, f"A clean golden failed: {clean_run.stderr}"
        assert noisy_run.returncode == 0, f"A noisy golden failed: {noisy_run.stderr}"
        clean_score = _extract_final_score(clean_run.stdout)
        noisy_score = _extract_final_score(noisy_run.stdout)
        assert clean_score > 95.0, f"A clean-as-pred final_score={clean_score:.2f}"
        assert noisy_score < 5.0, f"A noisy-as-pred final_score={noisy_score:.2f}"
        print(
            f"[PASS] test_a_board_evaluate_mock_golden_contract "
            f"clean={clean_score:.2f} noisy={noisy_score:.2f}"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _load_evaluator_module(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_triangle_distance_fallback_geometry():
    """无可选几何库时，P2S fallback 仍须计算点到三角面的精确距离."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    c = np.array([0.0, 1.0, 0.0])
    points = np.array([
        [0.2, 0.2, 0.0],   # 面内
        [0.2, 0.2, 2.0],   # 垂直于面
        [-1.0, 0.0, 0.0],  # 顶点最近
        [1.0, 1.0, 0.0],   # 斜边最近
    ])
    expected = np.array([0.0, 4.0, 1.0, 0.5])
    evaluator_paths = {
        "b": os.path.join(_STARTER, "evaluate_mock.py"),
        "a": os.path.join(
            _STARTER, "..", "..", "a_board", "starter_code", "evaluate_mock.py"
        ),
    }
    for stage, evaluator_path in evaluator_paths.items():
        evaluator = _load_evaluator_module(evaluator_path, f"mock_evaluator_{stage}")
        actual = evaluator._point_to_triangle_squared_distance(points, a, b, c)
        np.testing.assert_allclose(actual, expected, atol=1e-12)
    print("[PASS] test_triangle_distance_fallback_geometry")


def test_evaluate_mock_missing_mesh_fail_loudly():
    """缺 mesh/norm 默认 fail-loudly, 不静默 skip."""
    tmp = _build_golden_fixture()
    try:
        # 删掉一个样本的 mesh 目录, 模拟缺 mesh.
        import shutil as _shutil
        victim = os.path.join(tmp, "dataset", "train", "shapenet", "0440", "sample_0")
        _shutil.rmtree(victim)

        r = _run_evaluate_mock(tmp, pred_filename="clean.npy")
        assert r.returncode != 0, "should fail-loudly on missing mesh"
        combined = r.stdout + "\n" + r.stderr
        assert "missing mesh" in combined.lower() or "[FAIL]" in combined, combined
        print("[PASS] test_evaluate_mock_missing_mesh_fail_loudly")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluate_mock_missing_mesh_allow_flag():
    """--allow-missing-mesh-norm 时, 缺 mesh 的样本记 0 分继续."""
    tmp = _build_golden_fixture()
    try:
        import shutil as _shutil
        _shutil.rmtree(os.path.join(tmp, "dataset", "train", "shapenet", "0440", "sample_0"))
        # 直接调 evaluate_mock (不经 run_mock_eval)
        evaluate_mock = os.path.join(_STARTER, "evaluate_mock.py")
        cmd = [
            sys.executable, evaluate_mock,
            "--pred_dir", os.path.join(tmp, "dataset", "mock_test"),
            "--gt_dir", os.path.join(tmp, "dataset", "mock_test"),
            "--noisy_dir", os.path.join(tmp, "dataset", "mock_test"),
            "--mesh_dir", os.path.join(tmp, "dataset", "train"),
            "--pred_filename", "clean.npy",
            "--workers", "1",
            "--allow-missing-mesh-norm",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, f"with --allow should succeed: {r.stderr}"
        # sample_0 被跳过记 0, sample_1 正常得 100 -> 平均应在 50 附近
        final = _extract_final_score(r.stdout)
        assert 30 <= final <= 60, (
            f"expected ~50 (one missing + one perfect), got {final}.\n"
            f"stdout:\n{r.stdout}"
        )
        print(f"[PASS] test_evaluate_mock_missing_mesh_allow_flag  final={final:.2f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_eval_refuses_unknown_verdict():
    """非 golden 时 verdict=unknown 默认拒绝 (manifest 缺 summary.verdict)."""
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="green", n_samples=3)
        # 删 summary.verdict 字段, 模拟 unknown
        mani_path = os.path.join(predict_run, "manifest.json")
        with open(mani_path, "r", encoding="utf-8") as f:
            mani = json.load(f)
        del mani["summary"]["verdict"]
        with open(mani_path, "w", encoding="utf-8") as f:
            json.dump(mani, f, indent=2)

        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        r = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_unknown",
        ])
        assert r.returncode != 0, f"should refuse unknown verdict: {r.stdout}"
        # --force 能过
        r_force = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_unknown_force",
            "--force",
        ])
        assert r_force.returncode == 0, f"--force should succeed: {r_force.stderr}"
        print("[PASS] test_run_eval_refuses_unknown_verdict")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_eval_propagates_evaluator_returncode():
    """evaluator 失败时 run_mock_eval 必须也返回非零."""
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="green", n_samples=3)
        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        # 把 fake evaluator 换成一个一定失败的版本
        failing_eval = textwrap.dedent('''
            #!/usr/bin/env python3
            import sys
            print("fake failure from test")
            sys.exit(42)
        ''').lstrip()
        with open(os.path.join(root, "starter_code", "evaluate_mock.py"), "w",
                  encoding="utf-8") as f:
            f.write(failing_eval)

        r = _run_mock_eval(root, [
            "--predict-run", predict_run,
            "--mock-dir", "dataset/mock_test",
            "--mesh-dir", "dataset/train",
            "--eval-root", os.path.join(root, "outputs", "evals"),
            "--eval-id", "test_eval_fail_propagation",
        ])
        assert r.returncode == 42, (
            f"expected returncode=42 propagated from evaluator, got {r.returncode}\n"
            f"stdout: {r.stdout}"
        )
        # manifest 仍应生成 (失败仍写日志)
        mani = os.path.join(root, "outputs", "evals",
                            "b_final", "test_eval_fail_propagation", "manifest.json")
        assert os.path.exists(mani)
        print("[PASS] test_run_eval_propagates_evaluator_returncode")
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_run_eval_resolves_relative_paths_from_project_root():
    """--eval-root 等相对路径从项目根解析, 不依赖 CWD."""
    root = _make_synthetic_pdlts_root()
    try:
        _make_mock_data_dir(root, n_samples=3)
        predict_run = _make_mock_predict_run(root, verdict="green", n_samples=3)
        mesh_dir = os.path.join(root, "dataset", "train")
        os.makedirs(mesh_dir, exist_ok=True)

        # 故意从 root 之外的随机 CWD 调脚本, 看 --eval-root "outputs/evals" 能否
        # 落到 root/outputs/evals (而非 CWD/outputs/evals).
        other_cwd = tempfile.mkdtemp(prefix="mock_eval_other_cwd_")
        try:
            script = os.path.join(root, "scripts", "shared", "run_mock_eval.py")
            cmd = [
                sys.executable, script,
                "--predict-run", predict_run,
                "--mock-dir", os.path.join(root, "dataset", "mock_test"),
                "--mesh-dir", os.path.join(root, "dataset", "train"),
                "--eval-root", "outputs/evals",  # 相对路径
                "--eval-id", "test_eval_path_resolve",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=other_cwd)
            assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
            # 产物必须在 root/outputs/evals 下, 不是 other_cwd/outputs/evals
            expected = os.path.join(root, "outputs", "evals", "b_final",
                                    "test_eval_path_resolve", "manifest.json")
            unexpected = os.path.join(other_cwd, "outputs", "evals",
                                      "test_eval_path_resolve", "manifest.json")
            assert os.path.exists(expected), f"missing {expected}"
            assert not os.path.exists(unexpected), (
                f"path resolved against CWD, not project root: {unexpected}"
            )
            print("[PASS] test_run_eval_resolves_relative_paths_from_project_root")
        finally:
            shutil.rmtree(other_cwd, ignore_errors=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    print("=" * 60)
    print("mock eval pipeline test")
    print("=" * 60)
    # A. build-datalist
    test_build_datalist_full()
    test_build_datalist_limit()
    # B. run_mock_eval (fake evaluator)
    test_run_eval_green_happy_path()
    test_run_eval_refuses_empty_mesh_dir()
    test_run_eval_golden_mode_clean_as_pred()
    test_run_eval_refuses_red_verdict()
    test_run_eval_refuses_unknown_verdict()
    test_run_eval_force_override_red()
    test_run_eval_propagates_evaluator_returncode()
    test_run_eval_resolves_relative_paths_from_project_root()
    # C. 金标准 (真 evaluate_mock.py)
    test_evaluate_mock_golden_clean_as_pred()
    test_evaluate_mock_golden_noisy_as_pred()
    test_a_board_evaluate_mock_golden_contract()
    test_triangle_distance_fallback_geometry()
    test_evaluate_mock_missing_mesh_fail_loudly()
    test_evaluate_mock_missing_mesh_allow_flag()
    print("=" * 60)
    print("ALL PDLTS LIGHT MOCK EVALUATION TESTS PASSED")
    print("=" * 60)
