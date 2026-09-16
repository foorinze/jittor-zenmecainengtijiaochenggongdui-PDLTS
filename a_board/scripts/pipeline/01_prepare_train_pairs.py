#!/usr/bin/env python3
"""从原始网格生成第二阶段训练所需的固定 noisy/clean 点云对。

官方训练集只给网格，不给点云。第一阶段训练是每个 epoch 在线采样 + 在线加噪，
不需要这一步。但第二阶段需要固定的真值点云做监督目标——如果每个 epoch 重新
采样真值，那么「第一阶段输出」与「真值」就不是同一次采样的结果，逐点对应
关系不成立，残余映射也就无从学习。

因此这一步把采样与加噪固定下来，落盘为：

    <out_root>/<rel>/clean.npy   网格表面采样得到的干净点云
    <out_root>/<rel>/noisy.npy   加噪后的点云
    <out_root>/<rel>/norm.json   该样本的归一化参数与噪声强度记录

采样与噪声的随机种子逐样本固定（sample_seed = seed + index * seed_stride），
同一命令重跑得到逐位相同的结果。

噪声与第一阶段训练同口径：先归一化到单位球，再加 Laplace 噪声，逐样本的
噪声强度在 [0.005, 0.020] 内均匀抽取。

本脚本只做数据落盘，不训练，不切 patch。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import trimesh


VERSION = "a_final"
SCRIPT_VERSION = "public"
STAGE = "prepare_train_pairs"


def _find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "starter_code").is_dir() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("cannot locate PDLTS repository root")


ROOT = _find_repo_root()
STARTER = ROOT / "starter_code"
if str(STARTER) not in sys.path:
    sys.path.insert(0, str(STARTER))

from src.data.asset import Asset  # noqa: E402
from src.data.augment import AugmentSample  # noqa: E402


def _resolve(path: Union[str, Path]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_rels(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        rels = [line.strip() for line in f if line.strip()]
    if not rels:
        raise RuntimeError(f"empty datalist: {path}")
    return rels


def _class_of(rel: str) -> str:
    parts = rel.split("/")
    return parts[1] if len(parts) > 1 else parts[0]


def _select_rels(rels: Sequence[str], samples: int, policy: str, seed: int) -> List[str]:
    if samples <= 0:
        raise ValueError("--samples must be positive")
    if samples > len(rels):
        raise ValueError(f"--samples={samples} exceeds datalist size {len(rels)}")

    if policy == "sequential":
        selected = list(rels[:samples])
    elif policy == "random":
        rng = np.random.RandomState(int(seed))
        idx = rng.permutation(len(rels))[:samples]
        selected = [rels[i] for i in sorted(idx)]
    elif policy == "stratified":
        rng = np.random.RandomState(int(seed))
        by_cls: Dict[str, List[str]] = {}
        for rel in rels:
            by_cls.setdefault(_class_of(rel), []).append(rel)
        for cls in sorted(by_cls):
            rng.shuffle(by_cls[cls])
        classes = sorted(by_cls)
        selected = []
        cursor = 0
        while len(selected) < samples and any(cursor < len(by_cls[c]) for c in classes):
            for cls in classes:
                if len(selected) >= samples:
                    break
                if cursor < len(by_cls[cls]):
                    selected.append(by_cls[cls][cursor])
            cursor += 1
    else:
        raise ValueError(f"unknown sample policy: {policy}")

    if len(selected) != samples:
        raise RuntimeError(f"selected {len(selected)} samples, expected {samples}")
    if len(set(selected)) != len(selected):
        raise RuntimeError("selected datalist contains duplicates")
    return selected


def _bbox_center_scale(pc: np.ndarray) -> Tuple[np.ndarray, float]:
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = ((p_max + p_min) * 0.5).astype(np.float64)
    shifted = pc - center
    scale = float(np.sqrt((shifted * shifted).sum(axis=1).max()))
    if scale < 1e-12:
        raise ValueError("invalid bbox scale")
    return center, scale


def _sha1_array(arr: np.ndarray) -> str:
    arr32 = np.ascontiguousarray(arr.astype(np.float32, copy=False))
    return hashlib.sha1(arr32.tobytes()).hexdigest()


def _load_asset(mesh_path: Path, cls_name: str) -> Asset:
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return Asset(
        path=str(mesh_path),
        cls=cls_name,
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.faces),
    )


def _class_counts(rels: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rel in rels:
        cls = _class_of(rel)
        counts[cls] = counts.get(cls, 0) + 1
    return dict(sorted(counts.items()))


def _write_datalist(path: Path, rels: Sequence[str], overwrite: bool) -> None:
    payload = "".join(f"{rel}\n" for rel in rels)
    if path.exists() and not overwrite:
        existing = path.read_text(encoding="utf-8")
        if existing != payload:
            raise FileExistsError(f"datalist exists with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _sample_dir(out_root: Path, rel: str) -> Path:
    return out_root / rel


def _sample_complete(sample_dir: Path) -> bool:
    return all(
        (sample_dir / name).is_file()
        for name in ("clean.npy", "noisy.npy", "norm.json", "meta.json")
    )


def _existing_row(rel: str, index: int, sample_dir: Path) -> Dict:
    norm = _read_json(sample_dir / "norm.json")
    return {
        "rel": rel,
        "cls": _class_of(rel),
        "sample_index": int(index),
        "status": "skipped_existing",
        "mesh_path": norm.get("mesh_path", ""),
        "sample_seed": norm.get("sample_seed"),
        "noise_std": norm.get("noise", {}).get("std"),
        "clean_path": str(sample_dir / "clean.npy"),
        "noisy_path": str(sample_dir / "noisy.npy"),
        "clean_sha1_float32": norm.get("clean_sha1_float32"),
        "noisy_sha1_float32": norm.get("noisy_sha1_float32"),
    }


def _materialize_one(rel: str, index: int, args: argparse.Namespace, out_root: Path) -> Dict:
    sample_dir = _sample_dir(out_root, rel)
    if args.resume and not args.overwrite and _sample_complete(sample_dir):
        return _existing_row(rel, index, sample_dir)

    cls = _class_of(rel)
    mesh_path = _resolve(args.train_root) / rel / args.mesh_name
    if not mesh_path.exists():
        raise FileNotFoundError(f"mesh missing for {rel}: {mesh_path}")

    sample_seed = int(args.seed) + index * int(args.seed_stride)
    np.random.seed(sample_seed)

    asset = _load_asset(mesh_path, cls)
    AugmentSample(
        num_samples=int(args.n_points),
        num_vertex_samples=int(args.num_vertex_samples),
    ).apply(asset)
    if asset.sampled_vertices is None:
        raise RuntimeError(f"sampling produced no points for {rel}")
    raw_clean = np.asarray(asset.sampled_vertices, dtype=np.float64)
    if raw_clean.shape != (int(args.n_points), 3):
        raise ValueError(f"unexpected clean shape for {rel}: {raw_clean.shape}")

    center, scale = _bbox_center_scale(raw_clean)
    clean = ((raw_clean - center) / scale).astype(np.float32)

    # Match the historical frozen-pair noise formula and record it explicitly.
    noise_std = float(np.random.uniform(float(args.noise_std_min), float(args.noise_std_max)))
    noisy = clean + np.random.laplace(0.0, noise_std, size=clean.shape).astype(np.float32)
    noisy = noisy.astype(np.float32)

    sample_dir.mkdir(parents=True, exist_ok=True)
    clean_path = sample_dir / "clean.npy"
    noisy_path = sample_dir / "noisy.npy"
    np.save(clean_path, np.ascontiguousarray(clean))
    np.save(noisy_path, np.ascontiguousarray(noisy))

    norm_payload = {
        "rel": rel,
        "mesh_path": str(mesh_path),
        "sample_index": int(index),
        "sample_seed": int(sample_seed),
        "seed_stride": int(args.seed_stride),
        "n_points": int(args.n_points),
        "num_vertex_samples": int(args.num_vertex_samples),
        "normalization": "clean_bbox_center_max_radius",
        "center": center.astype(float).tolist(),
        "scale": float(scale),
        "noise": {
            "type": "laplace",
            "std": noise_std,
            "std_min": float(args.noise_std_min),
            "std_max": float(args.noise_std_max),
        },
        "clean_sha1_float32": _sha1_array(clean),
        "noisy_sha1_float32": _sha1_array(noisy),
        "script_version": SCRIPT_VERSION,
        "version": VERSION,
        "stage": STAGE,
    }
    _write_json(sample_dir / "norm.json", norm_payload)
    _write_json(sample_dir / "meta.json", norm_payload)

    return {
        "rel": rel,
        "cls": cls,
        "sample_index": int(index),
        "status": "written",
        "mesh_path": str(mesh_path),
        "sample_seed": int(sample_seed),
        "noise_std": noise_std,
        "clean_path": str(clean_path),
        "noisy_path": str(noisy_path),
        "clean_sha1_float32": norm_payload["clean_sha1_float32"],
        "noisy_sha1_float32": norm_payload["noisy_sha1_float32"],
    }


def cmd_prepare(args: argparse.Namespace) -> int:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")

    source_datalist = _resolve(args.source_datalist)
    datalist_out = _resolve(args.datalist_out)
    out_root = _resolve(args.out_root)
    mock_datalist = _resolve(args.mock_datalist)

    rels = _read_rels(source_datalist)
    selected = _select_rels(
        rels,
        samples=int(args.samples),
        policy=str(args.sample_policy),
        seed=int(args.seed),
    )

    if mock_datalist.exists() and not args.allow_mock_overlap:
        mock_rels = set(_read_rels(mock_datalist))
        overlap = sorted(set(selected) & mock_rels)
        if overlap:
            raise RuntimeError(
                "selected train pairs overlap mock datalist: "
                + ", ".join(overlap[:10])
            )

    missing_meshes = [
        rel for rel in selected
        if not (_resolve(args.train_root) / rel / args.mesh_name).exists()
    ]
    if missing_meshes:
        raise FileNotFoundError(
            "selected rels have missing meshes: "
            + ", ".join(missing_meshes[:10])
        )

    artifact_id = f"{_timestamp()}_{VERSION}_{STAGE}_n{len(selected)}"
    diag_dir = ROOT / "outputs" / "diagnostics" / VERSION / artifact_id
    diag_dir.mkdir(parents=True, exist_ok=True)
    (diag_dir / "command.sh").write_text(
        "#!/usr/bin/env bash\n" + " ".join(sys.argv) + "\n",
        encoding="utf-8",
    )
    (diag_dir / "selected_rels.txt").write_text(
        "".join(f"{rel}\n" for rel in selected),
        encoding="utf-8",
    )

    if args.dry_run:
        rows: List[Dict] = []
    else:
        if out_root.exists() and any(out_root.iterdir()) and not (args.overwrite or args.resume):
            raise FileExistsError(f"out root is non-empty; pass --overwrite or --resume: {out_root}")
        _write_datalist(datalist_out, selected, overwrite=bool(args.overwrite or args.resume))
        rows = []
        for i, rel in enumerate(selected):
            row = _materialize_one(rel, i, args, out_root)
            rows.append(row)
            if (i + 1) % max(1, int(args.log_every)) == 0 or i == 0 or i + 1 == len(selected):
                print(
                    f"[train-pairs] {i + 1}/{len(selected)} {rel} "
                    f"status={row['status']} noise_std={row.get('noise_std')}",
                    flush=True,
                )

    n_written = sum(1 for row in rows if row.get("status") == "written")
    n_skipped = sum(1 for row in rows if row.get("status") == "skipped_existing")
    manifest = {
        "version": VERSION,
        "script_version": SCRIPT_VERSION,
        "stage": STAGE,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_id": artifact_id,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "source_datalist": str(source_datalist),
        "datalist_out": str(datalist_out),
        "out_root": str(out_root),
        "train_root": str(_resolve(args.train_root)),
        "mesh_name": str(args.mesh_name),
        "samples": len(selected),
        "sample_policy": str(args.sample_policy),
        "seed": int(args.seed),
        "seed_stride": int(args.seed_stride),
        "n_points": int(args.n_points),
        "num_vertex_samples": int(args.num_vertex_samples),
        "noise_std_min": float(args.noise_std_min),
        "noise_std_max": float(args.noise_std_max),
        "selected_class_counts": _class_counts(selected),
        "missing_mesh_count": 0,
        "n_written": int(n_written),
        "n_skipped_existing": int(n_skipped),
        "materialized_rows": rows,
        "command": sys.argv,
    }
    _write_json(diag_dir / "manifest.json", manifest)
    summary = {
        **manifest,
        "diagnostic_dir": str(diag_dir),
        "n_materialized": int(n_written + n_skipped),
        "materialized_class_counts": _class_counts([row["rel"] for row in rows]),
    }
    _write_json(diag_dir / "summary.json", summary)
    print(f"[train-pairs] summary: {diag_dir}", flush=True)
    if args.dry_run:
        print("[train-pairs] dry run only; no datalist/data files were written", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="从原始网格生成固定 noisy/clean 点云对")
    p.add_argument("--source-datalist", default="starter_code/datalist/train_full15k.txt")
    p.add_argument("--datalist-out", default="starter_code/datalist/train_full15k_generated.txt")
    p.add_argument("--train-root", default="dataset/train")
    p.add_argument("--out-root", default="dataset/a_final_train_full15k")
    p.add_argument("--mock-datalist", default="starter_code/datalist/mock_full.txt")
    p.add_argument("--mesh-name", default="models/model_normalized.obj")
    p.add_argument("--samples", type=int, default=15732)
    p.add_argument("--sample-policy", choices=("stratified", "random", "sequential"), default="sequential")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--seed-stride", type=int, default=9973)
    p.add_argument("--n-points", type=int, default=50000)
    p.add_argument("--num-vertex-samples", type=int, default=1024)
    p.add_argument("--noise-std-min", type=float, default=0.005)
    p.add_argument("--noise-std-max", type=float, default=0.020)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    # 本地验证集是从 dataset/train 构造的，训练集全量与它天然有交集。
    # 这一步只是把训练集全量的网格采样落盘，不涉及子集选择，所以允许重叠；
    # 第二阶段的训练子集会在 02 步显式剔除全部交集。
    p.add_argument("--allow-mock-overlap", action="store_true")
    return cmd_prepare(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
