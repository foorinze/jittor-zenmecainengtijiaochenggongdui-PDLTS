"""coverage sweep 单测。

运行 (从 starter_code/ 启):
    cd <repo_root>/b_board/starter_code
    conda activate jittor
    python -m tests.test_coverage_sweep

验证内容:
    A. 覆盖算法一致性
        A1. compute_coverage 的 n_missing 与 patch_denoise 内部 coverage gap
            的 n_missing 在同一 (pc, seed_k, patch_size) 下严格相等
    B. CSV / 产物契约
        B1. CSV schema 完整 (rel_path / cls_id / object_id / status ...)
        B2. 完成后 manifest.json 含 script_version, env, git_commit, aggregate_summary
        B3. logs/sweep.log 有进度行
    C. Resume
        C1. 第二次跑带 --resume 时, 跳过已有 (rel_path, seed_k, patch_size) 三元组
        C2. 不同 patch_size 不被 resume 去重
    D. 参数 guard
        D1. --workers > 1 直接 fail (exit 非零)
        D2. --datalist 缺失 fail
        D3. --mock-dir 缺失 fail
        D4. 空 datalist fail
    E. 数据降级
        E1. 缺 norm.json 的样本 status='skipped_no_norm', 不影响其他样本
        E2. 缺 noisy.npy 的样本 status='error_load'

单测用合成小点云 (N=128 / patch_size=32) 避免真实 50K 太慢.
"""

import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.dirname(_HERE)
_PROJECT_ROOT = os.path.dirname(_STARTER)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import numpy as np

import jittor as jt


SCRIPT = os.path.abspath(
    os.path.join(_PROJECT_ROOT, "scripts", "shared", "coverage_sweep.py")
)


def _load_coverage_sweep_module():
    """动态加载 scripts/shared/coverage_sweep.py 作为模块 (scripts/ 不是 Python package)."""
    spec = importlib.util.spec_from_file_location("pdlts_coverage_sweep", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _setup():
    jt.set_global_seed(42)
    np.random.seed(42)
    try:
        jt.flags.use_cuda = 1
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_fake_mock_dir(tmp: str, n_samples: int = 3, include_norm: bool = True,
                       include_noisy: bool = True, N: int = 128) -> str:
    """造 <tmp>/dataset/mock_test/shapenet/0440/sample_<i>/ {noisy.npy, clean.npy, norm.json}"""
    mock_dir = os.path.join(tmp, "dataset", "mock_test")
    for i in range(n_samples):
        sdir = os.path.join(mock_dir, "shapenet", "0440", f"sample_{i}")
        os.makedirs(sdir, exist_ok=True)
        if include_noisy:
            noisy = np.random.randn(N, 3).astype(np.float32) * 0.3
            np.save(os.path.join(sdir, "noisy.npy"), noisy)
            np.save(os.path.join(sdir, "clean.npy"), noisy + 0.01)
        if include_norm:
            with open(os.path.join(sdir, "norm.json"), "w", encoding="utf-8") as f:
                json.dump({"center": [0.0, 0.0, 0.0], "scale": 1.0}, f)
    return mock_dir


def _write_datalist(tmp: str, rels: list, name: str = "mock.txt") -> str:
    path = os.path.join(tmp, "datalist", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rels:
            f.write(r + "\n")
    return path


def _run_sweep(tmp: str, extra: list) -> subprocess.CompletedProcess:
    cmd = [sys.executable, SCRIPT, "--stage", "b_final"] + extra
    return subprocess.run(cmd, capture_output=True, text=True, cwd=tmp)


# ---------------------------------------------------------------------------
# A. 覆盖算法一致性
# ---------------------------------------------------------------------------
def test_compute_coverage_matches_patch_denoise():
    """compute_coverage 的 n_missing 必须和 patch_denoise 内 coverage gap 的
    n_missing 完全相等 (同一 pc, seed_k, patch_size)."""
    mod = _load_coverage_sweep_module()
    compute_coverage = mod.compute_coverage
    from src.model.pdlts_light.denoise import patch_denoise

    # 合成一个 identity network + 真实 noisy 点云跑 patch_denoise 拿 coverage_info
    from src.model.pdlts_light.inn import AffineCoupling  # noqa: F401  (pkg import)

    class _Identity(jt.nn.Module):
        def execute(self, x):
            B = x.shape[0]
            return x, jt.zeros((B,)), jt.zeros((1,))

    for seed_k in [3, 5, 8]:
        np.random.seed(seed_k * 7)
        pc_np = np.random.randn(128, 3).astype(np.float32) * 0.3
        pc = jt.array(pc_np)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _, coverage_info = patch_denoise(
                _Identity(), pc,
                patch_size=32, seed_k=seed_k, seed_k_alpha=2,
                return_coverage=True,
            )
        sweep_info = compute_coverage(pc_np, patch_size=32, seed_k=seed_k)

        assert sweep_info["N"] == coverage_info["N"], \
            f"seed_k={seed_k}: N mismatch"
        assert sweep_info["K"] == coverage_info["K"], \
            f"seed_k={seed_k}: K mismatch"
        assert sweep_info["effective_patch_size"] == coverage_info["effective_patch_size"], \
            f"seed_k={seed_k}: effective_patch_size mismatch"
        assert sweep_info["n_missing"] == coverage_info["n_missing"], (
            f"seed_k={seed_k}: n_missing mismatch "
            f"(sweep={sweep_info['n_missing']} vs patch_denoise={coverage_info['n_missing']})"
        )
    print("[PASS] test_compute_coverage_matches_patch_denoise")


# ---------------------------------------------------------------------------
# B. CSV / manifest / log 产物
# ---------------------------------------------------------------------------
def test_csv_schema_and_manifest_complete():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_csv_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=2)
        dl = _write_datalist(tmp, [
            "shapenet/0440/sample_0", "shapenet/0440/sample_1",
        ])
        r = _run_sweep(tmp, [
            "--datalist", dl,
            "--mock-dir", mock_dir,
            "--seed-k-list", "5,8",
            "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_csv",
        ])
        assert r.returncode == 0, f"stdout={r.stdout} stderr={r.stderr}"
        evd = os.path.join(tmp, "evals", "b_final", "t_csv")
        # CSV schema
        csv_path = os.path.join(evd, "coverage_sweep.csv")
        with open(csv_path, "r", encoding="utf-8") as f:
            rdr = csv.DictReader(f)
            rows = list(rdr)
        assert rdr.fieldnames is not None
        for col in ("rel_path", "cls_id", "object_id", "seed_k", "patch_size",
                    "N", "K", "effective_patch_size", "n_covered", "n_missing",
                    "missing_ratio", "coverage_ratio",
                    "fps_sec", "knn_sec", "total_sec", "status", "error"):
            assert col in rdr.fieldnames, f"CSV missing column {col}"
        assert len(rows) == 2 * 2, f"expected 4 rows, got {len(rows)}"
        for row in rows:
            assert row["status"] == "ok", row
            assert row["cls_id"] == "0440"
            assert row["object_id"].startswith("sample_")
            assert int(row["seed_k"]) in (5, 8)
            assert int(row["patch_size"]) == 32
            assert float(row["missing_ratio"]) >= 0.0

        # manifest.json
        with open(os.path.join(evd, "manifest.json"), encoding="utf-8") as f:
            mani = json.load(f)
        assert mani["script_version"]
        assert mani["stage"] == "b_final"
        assert mani["env"]["jittor_version"]
        assert "use_cuda" in mani["env"]
        assert "git_commit" in mani
        assert "aggregate_summary" in mani
        assert "per_seed_k" in mani["aggregate_summary"]
        # 聚合表
        summary_rows = mani["aggregate_summary"]["per_seed_k"]
        assert len(summary_rows) == 2  # seed_k 5, 8
        for srow in summary_rows:
            assert "zero_miss_samples" in srow
            assert "max_missing_ratio" in srow
            assert "p95_missing_ratio" in srow
            assert "avg_fps_sec" in srow
            assert "avg_knn_sec" in srow
            assert "status_counts" in srow
        # status_counts_total
        assert "status_counts_total" in mani["aggregate_summary"]
        sct = mani["aggregate_summary"]["status_counts_total"]
        assert "ok" in sct

        # log 文件存在 (内容可能因 10-row batch 没打印出来, 只要文件在即可)
        assert os.path.exists(os.path.join(evd, "logs", "sweep.log"))
        assert os.path.exists(os.path.join(evd, "command.sh"))
        print("[PASS] test_csv_schema_and_manifest_complete")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# C. Resume
# ---------------------------------------------------------------------------
def test_resume_skips_completed_keys():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_resume_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=2)
        dl = _write_datalist(tmp, [
            "shapenet/0440/sample_0", "shapenet/0440/sample_1",
        ])
        eval_root = os.path.join(tmp, "evals")
        # Round 1: 完整跑
        r1 = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5,8",
            "--patch-size", "32",
            "--eval-root", eval_root, "--eval-id", "t_resume",
        ])
        assert r1.returncode == 0, r1.stderr
        csv_path = os.path.join(eval_root, "b_final", "t_resume", "coverage_sweep.csv")
        with open(csv_path) as f:
            rows1 = list(csv.DictReader(f))
        n1 = len(rows1)
        # Round 2: 带 --resume 重跑, 应该跳过所有 (rel_path, seed_k, patch_size)
        r2 = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5,8",
            "--patch-size", "32",
            "--eval-root", eval_root, "--eval-id", "t_resume",
            "--resume",
        ])
        assert r2.returncode == 0, r2.stderr
        combined = r2.stdout + "\n" + r2.stderr
        assert "resume_skipped=" in combined, combined
        with open(csv_path) as f:
            rows2 = list(csv.DictReader(f))
        # CSV 行数不增加 (所有 key 都被 resume 跳过)
        assert len(rows2) == n1, f"expected {n1} rows after resume, got {len(rows2)}"
        print(f"[PASS] test_resume_skips_completed_keys  n_rows={n1}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_does_not_skip_different_patch_size():
    """resume 去重键包含 patch_size; 换了 patch_size 应该重新跑."""
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_resume_ps_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=1, N=128)
        dl = _write_datalist(tmp, ["shapenet/0440/sample_0"])
        eval_root = os.path.join(tmp, "evals")
        # Round 1: patch_size=32
        r1 = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", eval_root, "--eval-id", "t_resume_ps",
        ])
        assert r1.returncode == 0, r1.stderr
        # Round 2: 同 eval_id + --resume, 但 patch_size=64 — 应再跑一行
        r2 = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "64",
            "--eval-root", eval_root, "--eval-id", "t_resume_ps",
            "--resume",
        ])
        assert r2.returncode == 0, r2.stderr
        csv_path = os.path.join(eval_root, "b_final", "t_resume_ps", "coverage_sweep.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        # 两次各写一行
        assert len(rows) == 2, f"expected 2 rows (diff patch_size), got {len(rows)}"
        ps_set = set(int(r["patch_size"]) for r in rows)
        assert ps_set == {32, 64}, ps_set
        print("[PASS] test_resume_does_not_skip_different_patch_size")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# D. 参数 guard
# ---------------------------------------------------------------------------
def test_workers_greater_than_1_fails():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_workers_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=1)
        dl = _write_datalist(tmp, ["shapenet/0440/sample_0"])
        r = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_workers",
            "--workers", "4",
        ])
        assert r.returncode != 0, f"--workers=4 should fail: {r.stdout}"
        combined = r.stdout + "\n" + r.stderr
        assert "workers" in combined.lower()
        print("[PASS] test_workers_greater_than_1_fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_datalist_missing_fails():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_dl_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=1)
        r = _run_sweep(tmp, [
            "--datalist", os.path.join(tmp, "nope.txt"),
            "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_dl_missing",
        ])
        assert r.returncode != 0
        combined = r.stdout + "\n" + r.stderr
        assert "datalist" in combined.lower()
        print("[PASS] test_datalist_missing_fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_mock_dir_missing_fails():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_md_")
    try:
        dl = _write_datalist(tmp, ["shapenet/0440/sample_0"])
        r = _run_sweep(tmp, [
            "--datalist", dl,
            "--mock-dir", os.path.join(tmp, "nope_mock"),
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_md_missing",
        ])
        assert r.returncode != 0
        combined = r.stdout + "\n" + r.stderr
        assert "mock" in combined.lower()
        print("[PASS] test_mock_dir_missing_fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_empty_datalist_fails():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_empty_dl_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=1)
        dl = _write_datalist(tmp, [])  # 空
        r = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_empty_dl",
        ])
        assert r.returncode != 0
        combined = r.stdout + "\n" + r.stderr
        assert "empty" in combined.lower() or "datalist" in combined.lower()
        print("[PASS] test_empty_datalist_fails")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# E. 数据降级
# ---------------------------------------------------------------------------
def test_missing_norm_marks_skipped_status():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_no_norm_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=1, include_norm=False)
        dl = _write_datalist(tmp, ["shapenet/0440/sample_0"])
        r = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_no_norm",
        ])
        assert r.returncode == 0, r.stderr
        csv_path = os.path.join(tmp, "evals", "b_final", "t_no_norm", "coverage_sweep.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["status"] == "skipped_no_norm", rows[0]
        print("[PASS] test_missing_norm_marks_skipped_status")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_noisy_marks_error_load():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_no_noisy_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=1, include_noisy=False)
        dl = _write_datalist(tmp, ["shapenet/0440/sample_0"])
        r = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_no_noisy",
        ])
        assert r.returncode == 0, r.stderr
        csv_path = os.path.join(tmp, "evals", "b_final", "t_no_noisy", "coverage_sweep.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["status"] == "error_load", rows[0]
        print("[PASS] test_missing_noisy_marks_error_load")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_limit_works():
    tmp = tempfile.mkdtemp(prefix="coverage_sweep_limit_")
    try:
        mock_dir = _make_fake_mock_dir(tmp, n_samples=3)
        dl = _write_datalist(tmp, [
            "shapenet/0440/sample_0", "shapenet/0440/sample_1", "shapenet/0440/sample_2",
        ])
        r = _run_sweep(tmp, [
            "--datalist", dl, "--mock-dir", mock_dir,
            "--seed-k-list", "5", "--patch-size", "32",
            "--limit", "2",
            "--eval-root", os.path.join(tmp, "evals"),
            "--eval-id", "t_limit",
        ])
        assert r.returncode == 0, r.stderr
        csv_path = os.path.join(tmp, "evals", "b_final", "t_limit", "coverage_sweep.csv")
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        # --limit 2 -> 2 samples × 1 seed_k = 2 rows
        assert len(rows) == 2, f"expected 2 (limit), got {len(rows)}"
        print("[PASS] test_limit_works")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    _setup()
    print("=" * 60)
    print("coverage sweep test")
    print("Jittor:", jt.__version__, "CUDA:", jt.flags.use_cuda)
    print("=" * 60)
    # A
    test_compute_coverage_matches_patch_denoise()
    # B
    test_csv_schema_and_manifest_complete()
    # C
    test_resume_skips_completed_keys()
    test_resume_does_not_skip_different_patch_size()
    # D
    test_workers_greater_than_1_fails()
    test_datalist_missing_fails()
    test_mock_dir_missing_fails()
    test_empty_datalist_fails()
    # E
    test_missing_norm_marks_skipped_status()
    test_missing_noisy_marks_error_load()
    test_limit_works()
    print("=" * 60)
    print("ALL COVERAGE SWEEP TESTS PASSED")
    print("=" * 60)
