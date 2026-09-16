#!/usr/bin/env python3
"""把推理输出打包成提交用的 result.zip，打包前做完整性校验。

输入是 outputs/predictions/<stage>/<run_id>/pred/ 下的
shapenet/<cls>/<id>/denoised.npy。

打包前逐条校验，任一条不过就中止，不产出半成品 zip：
    1. 样本集合与 dataset/test_noisy 完全一致，多一个少一个都算失败；
    2. 每个 denoised.npy 的形状与对应 noisy.npy 一致；
    3. dtype 为 float32，且没有 NaN / Inf；
    4. 推理阶段自己的完整性结论必须是 green 或 yellow；
    5. zip 内部路径严格是 shapenet/<cls>/<id>/denoised.npy，不含多余顶层目录。

第 5 条是提交格式的硬要求：多一层顶层目录会导致评测方读不到文件。

输出目录：
    outputs/submissions/<stage>/<submit_id>/
      result.zip
      check_shapes.txt   逐样本的形状检查结果
      zip_list.txt       zip 内部路径清单
      manifest.json      指回推理 run_id、样本数、校验结论
      notes.md           说明占位

路径约定：--dataset-test 与 --submit-root 的相对路径都相对 a_board/ 根目录解析，
不相对当前工作目录。这样从 starter_code/ 下执行也不会解析偏。

用法：
    python scripts/shared/pack_submission.py \
        --predict-run outputs/predictions/a_final/repro_specialist_official \
        --submit-id 20260802_a_final_submit
"""

import argparse
import json
import os
import sys
import time
import zipfile
from pathlib import Path


def _find_repo_root() -> Path:
    """从 scripts/shared/pack_submission.py 定位 a_board/ 根目录。"""
    return Path(__file__).resolve().parents[2]


_REPO_ROOT = _find_repo_root()
for _path in (_REPO_ROOT / "scripts" / "shared", _REPO_ROOT / "scripts", _REPO_ROOT / "starter_code"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
from output_layout import infer_output_version, versioned_artifact_dir


ALLOWED_VERDICTS_DEFAULT = {"green", "yellow"}
ALLOWED_VERDICTS_FORCE = {"green", "yellow", "red", "unknown"}


def _find_project_root() -> str:
    """Return the project root used for resolving relative CLI paths."""
    return str(_REPO_ROOT)


def _resolve_path(p: str, project_root: str) -> str:
    """若是绝对路径原样返回; 否则相对 project_root."""
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(project_root, p))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--predict-run", required=True,
        help="路径指向 outputs/predictions/<stage>/<run_id> 目录 (可为相对项目根或绝对)",
    )
    p.add_argument(
        "--submit-id", default=None,
        help="输出 submit run id. 默认 YYYYMMDD_HHMMSS_submission",
    )
    p.add_argument(
        "--submit-root", default="outputs/submissions",
        help="submissions 根目录 (相对项目根或绝对)",
    )
    p.add_argument(
        "--dataset-test", default="dataset/test_noisy",
        help="官方 test_noisy 根目录 (相对项目根或绝对)",
    )
    p.add_argument(
        "--skip-completeness-check", action="store_true",
        help="跳过 pred vs dataset_test 样本集合一致性校验 (调试用, 仅限 mock eval)",
    )
    p.add_argument(
        "--force", action="store_true",
        help="即使 predict verdict=red 或 unknown 也强行打包 (调试用)",
    )
    return p.parse_args()


def _enumerate_pred_samples(pred_root: str) -> list:
    """返回 pred_root 下所有 denoised.npy 的 'shapenet/<cls>/<id>' 相对路径 (set-friendly list, sorted)."""
    keys = []
    shapenet_root = os.path.join(pred_root, "shapenet")
    if not os.path.isdir(shapenet_root):
        return keys
    for cls in sorted(os.listdir(shapenet_root)):
        cls_dir = os.path.join(shapenet_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for sample in sorted(os.listdir(cls_dir)):
            sdir = os.path.join(cls_dir, sample)
            dn = os.path.join(sdir, "denoised.npy")
            if os.path.isfile(dn):
                keys.append(f"shapenet/{cls}/{sample}")
    return keys


def _enumerate_test_samples(dataset_test: str) -> list:
    """返回 dataset_test 下所有 noisy.npy 的 'shapenet/<cls>/<id>' 相对路径."""
    keys = []
    shapenet_root = os.path.join(dataset_test, "shapenet")
    if not os.path.isdir(shapenet_root):
        return keys
    for cls in sorted(os.listdir(shapenet_root)):
        cls_dir = os.path.join(shapenet_root, cls)
        if not os.path.isdir(cls_dir):
            continue
        for sample in sorted(os.listdir(cls_dir)):
            sdir = os.path.join(cls_dir, sample)
            nn = os.path.join(sdir, "noisy.npy")
            if os.path.isfile(nn):
                keys.append(f"shapenet/{cls}/{sample}")
    return keys


def main():
    args = parse_args()
    project_root = _find_project_root()

    # 路径离开 CWD, 走 project_root
    predict_run = _resolve_path(args.predict_run, project_root)
    dataset_test = _resolve_path(args.dataset_test, project_root)
    submit_root = _resolve_path(args.submit_root, project_root)

    if not os.path.isdir(predict_run):
        sys.exit(f"[FAIL] predict_run not a directory: {predict_run}")
    pred_root = os.path.join(predict_run, "pred")
    if not os.path.isdir(pred_root):
        sys.exit(f"[FAIL] missing pred dir: {pred_root}")
    shapenet_root = os.path.join(pred_root, "shapenet")
    if not os.path.isdir(shapenet_root):
        sys.exit(f"[FAIL] missing shapenet dir under pred: {shapenet_root}")

    # --- 1. verdict 校验 (green / yellow 默认通过, red / unknown 要 --force) ---
    predict_manifest_path = os.path.join(predict_run, "manifest.json")
    predict_manifest = {}
    if os.path.exists(predict_manifest_path):
        with open(predict_manifest_path, "r", encoding="utf-8") as f:
            predict_manifest = json.load(f)
    verdict = (predict_manifest.get("summary") or {}).get("verdict", "unknown")
    allowed = ALLOWED_VERDICTS_FORCE if args.force else ALLOWED_VERDICTS_DEFAULT
    if verdict not in allowed:
        sys.exit(
            f"[REFUSE] predict verdict='{verdict}' not in {sorted(allowed)}. "
            f"Use --force to override for debugging."
        )

    # --- 2. 样本集合完整性：缺一个样本就不该提交 ---
    pred_keys = set(_enumerate_pred_samples(pred_root))
    if not pred_keys:
        sys.exit(f"[FAIL] no denoised.npy found under {shapenet_root}")

    if not args.skip_completeness_check:
        if not os.path.isdir(dataset_test):
            sys.exit(
                f"[FAIL] dataset_test not a directory: {dataset_test}. "
                f"Use --skip-completeness-check only if intentionally packing a "
                f"partial set (e.g. mock subset); see --help."
            )
        test_keys = set(_enumerate_test_samples(dataset_test))
        if not test_keys:
            sys.exit(f"[FAIL] no noisy.npy found under {dataset_test}")
        missing_pred = test_keys - pred_keys
        extra_pred = pred_keys - test_keys
        if missing_pred or extra_pred:
            msg = [
                f"[FAIL] pred sample set != test_noisy sample set.",
                f"  predict_run = {predict_run}",
                f"  dataset_test = {dataset_test}",
                f"  n_test = {len(test_keys)}  n_pred = {len(pred_keys)}",
                f"  missing (in test but not pred): {len(missing_pred)}",
                f"  extra   (in pred but not test): {len(extra_pred)}",
            ]
            if missing_pred:
                msg.append("  first 5 missing:")
                for k in list(sorted(missing_pred))[:5]:
                    msg.append(f"    {k}")
            if extra_pred:
                msg.append("  first 5 extra:")
                for k in list(sorted(extra_pred))[:5]:
                    msg.append(f"    {k}")
            sys.exit("\n".join(msg))

    # --- 3. 按集合顺序构造 denoised_files 列表 ---
    denoised_files = [
        os.path.join(pred_root, k, "denoised.npy") for k in sorted(pred_keys)
    ]

    # 准备 submit 目录
    submit_id = args.submit_id or (
        time.strftime("%Y%m%d_%H%M%S") + "_submission"
    )
    if infer_output_version(submit_id) == "_unversioned":
        submit_version = os.path.basename(os.path.dirname(predict_run))
        submit_dir = os.path.abspath(os.path.join(submit_root, submit_version, submit_id))
    else:
        submit_dir = os.path.abspath(versioned_artifact_dir(submit_root, submit_id))
    os.makedirs(submit_dir, exist_ok=True)

    # --- 4. Shape guard (per-sample vs noisy) ---
    import numpy as np
    check_lines = []
    check_lines.append(f"# shape check for predict run: {predict_run}\n")
    check_lines.append(f"# checked_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    check_lines.append(f"# n_samples: {len(denoised_files)}\n")
    bad = 0

    for dn_path in denoised_files:
        rel = os.path.relpath(dn_path, pred_root)
        rel_dir = os.path.dirname(rel)
        noisy_path = os.path.join(dataset_test, rel_dir, "noisy.npy")

        try:
            dn = np.load(dn_path)
        except Exception as e:
            check_lines.append(f"[BAD] cannot load denoised: {rel}  err={e}\n")
            bad += 1
            continue

        if dn.dtype != np.float32:
            check_lines.append(
                f"[BAD] dtype mismatch: {rel}  got={dn.dtype}  want=float32\n"
            )
            bad += 1
            continue
        if dn.ndim != 2 or dn.shape[-1] != 3:
            check_lines.append(
                f"[BAD] shape not (N,3): {rel}  got={dn.shape}\n"
            )
            bad += 1
            continue
        if not np.isfinite(dn).all():
            check_lines.append(
                f"[BAD] non-finite values: {rel}  n_bad={(~np.isfinite(dn)).sum()}\n"
            )
            bad += 1
            continue

        if os.path.exists(noisy_path):
            try:
                ny = np.load(noisy_path)
            except Exception as e:
                check_lines.append(
                    f"[BAD] cannot load noisy: {noisy_path}  err={e}\n"
                )
                bad += 1
                continue
            if dn.shape != ny.shape:
                check_lines.append(
                    f"[BAD] shape mismatch: {rel}  "
                    f"denoised={dn.shape}  noisy={ny.shape}\n"
                )
                bad += 1
                continue
            check_lines.append(
                f"[OK]  {rel}  shape={dn.shape}  dtype={dn.dtype}\n"
            )
        else:
            # 没有 noisy 对照 (mock data 跨目录) — 仅在 skip-completeness-check 下会来
            check_lines.append(
                f"[OK-no-noisy] {rel}  shape={dn.shape}  dtype={dn.dtype}\n"
            )

    check_path = os.path.join(submit_dir, "check_shapes.txt")
    with open(check_path, "w", encoding="utf-8") as f:
        f.writelines(check_lines)

    if bad > 0:
        sys.exit(
            f"[FAIL] {bad}/{len(denoised_files)} samples failed shape guard. "
            f"See {check_path}"
        )

    # --- 5. 打包 zip: 内部路径 shapenet/<cls>/<id>/denoised.npy ---
    zip_path = os.path.join(submit_dir, "result.zip")
    zip_list_lines = []
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for dn_path in denoised_files:
            arcname = os.path.relpath(dn_path, pred_root).replace("\\", "/")
            zf.write(dn_path, arcname)
            zip_list_lines.append(arcname + "\n")
    with open(os.path.join(submit_dir, "zip_list.txt"), "w", encoding="utf-8") as f:
        f.writelines(zip_list_lines)

    # zip 内部健全性
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        bad_names = [n for n in names
                     if not (n.startswith("shapenet/") and n.endswith("/denoised.npy"))]
        if bad_names:
            sys.exit(
                f"[FAIL] zip contains non-conforming entries: {bad_names[:5]}"
            )

    # --- 6. manifest.json ---
    manifest = {
        "submit_id": submit_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "predict_run": predict_run,
        "predict_manifest_summary": predict_manifest.get("summary", {}),
        "n_samples": len(denoised_files),
        "shape_guard_bad": bad,
        "zip_path": zip_path,
        "verdict_at_pack": verdict,
        "force": bool(args.force),
        "completeness_check": not args.skip_completeness_check,
    }
    with open(os.path.join(submit_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    notes_path = os.path.join(submit_dir, "notes.md")
    if not os.path.exists(notes_path):
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write("# Notes\n\n")
            f.write(f"- submit_id: {submit_id}\n")
            f.write(f"- predict_run: {predict_run}\n")
            f.write(f"- verdict (inherited): {verdict}\n")
            f.write(f"- n_samples: {len(denoised_files)}\n")
            f.write("- conclusion: pending (fill after online evaluation)\n")

    size_mb = os.path.getsize(zip_path) / 1024 / 1024
    print(f"[OK] packed {len(denoised_files)} samples into {zip_path} "
          f"({size_mb:.2f} MB)")
    print(f"     submit_dir = {submit_dir}")
    print(f"     verdict    = {verdict}")


if __name__ == "__main__":
    main()
