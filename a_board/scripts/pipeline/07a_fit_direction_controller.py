#!/usr/bin/env python3
"""第三阶段方向场的系数拟合。

`07_apply_direction_controller.py` 里的 `COEFFICIENTS` 与 `BASIS_SCALE` 是冻结
常量，本脚本就是产出这两组常量的拟合过程，让第三阶段也能从原始训练数据独立跑出来。

拟合目标是**官方复合评分的上升方向**，不是单一几何距离的下降方向。对每个拟合形状，
把官方评分两项（CD 与 P2S）各自的梯度用该样本自身的 noisy 误差归一化后相加取反：

    direction = −(∇CD / CD_noisy) − (∇P2S / P2S_noisy)

每项各带一个开关：只有当该项当前仍有改善空间（0 < 自身误差 < noisy 误差，即评分
没有被 clamp 到 0 或 100）时才计入，否则该项置零。direction 再整体归一化到
RMS = 0.02，与 07_apply 施加位移时的剂量口径一致，得到无量纲的逐点目标。

然后在 10 个多尺度基上做岭回归（lambda = 100）：每个形状按固定种子抽 4096 个点，
52 个形状的样本汇总后一次求解。基先按各自在拟合集上的 RMS 归一化，这组 RMS 就是
`BASIS_SCALE`。

数据边界：
    输入 source     第二阶段在 52 个拟合形状上的输出
    训练集真值      dataset/a_final_train_full15k/<样本>/{clean,noisy}.npy、norm.json
    训练集网格      dataset/train/<样本>/models/model_normalized.obj（P2S 梯度需要）

三者都只来自训练集，与官方测试集无交集。推理阶段（07_apply）只读输入点云自身的
坐标，不读上面任何一项。

用法：
    # 拟合。source 不存在时，先用第二阶段权重在 52 个形状上推理一遍再拟合
    python scripts/pipeline/07a_fit_direction_controller.py \\
        --firstpass-cache outputs/predictions/a_final/repro_firstpass_cache_merged/pred \\
        --stage2-ckpt outputs/runs/a_final/repro_specialist_step3_scorenorm/specialist_step3_scorenorm.pkl

    # 已有第二阶段输出时直接拟合
    python scripts/pipeline/07a_fit_direction_controller.py --source <目录>

    # 与 07_apply 里的冻结值逐项比较
    python scripts/pipeline/07a_fit_direction_controller.py --source <目录> --verify-frozen
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
STARTER_CODE = REPO_ROOT / "starter_code"

VERSION = "a_final"
SCRIPT_VERSION = "public"

# 岭回归正则强度。
RIDGE_LAMBDA = 100.0
# 每个拟合形状抽多少个点进回归。50k 个点全用进去没有必要，也会让 Gram 矩阵累加
# 的浮点求和顺序依赖点数。
TRAIN_POINTS = 4096
# 抽点用的固定种子，逐形状偏移，保证可复现。
SAMPLE_SEED = 20_260_727
N_POINTS = 50_000

DEFAULT_SOURCE = (
    f"outputs/predictions/{VERSION}/repro_controller_fit_specialist/pred"
)
DEFAULT_DATALIST = "starter_code/datalist/controller_fit_shapes.txt"
DEFAULT_TRAIN_ROOT = "dataset/a_final_train_full15k"
DEFAULT_MESH_ROOT = "dataset/train"
# run.py 的 CWD 是 starter_code/，所以权重路径要以它为基准，带 ../
DEFAULT_STAGE2_CKPT = (
    f"../outputs/runs/{VERSION}/repro_specialist_step3_scorenorm/"
    "specialist_step3_scorenorm.pkl"
)
DEFAULT_OUT = f"outputs/runs/{VERSION}/repro_controller_fit"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def import_by_path(name: str, path: Path):
    """按文件路径导入模块。文件名以数字开头，不能用普通 import 语句。"""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 基向量场、方向合成、剂量归一化全部复用应用侧的实现，避免两份实现漂移。
APPLY = import_by_path(
    "apply_direction_controller", SCRIPT_DIR / "07_apply_direction_controller.py"
)
# 官方评分口径复用评测脚本本体，不另写一份 CD / P2S。
EVALUATOR = import_by_path("evaluate_mock", STARTER_CODE / "evaluate_mock.py")

K_VALUES = APPLY.K_VALUES
DOSE_OVER_H = APPLY.DOSE_OVER_H
multiscale_bases = APPLY.multiscale_bases
ridge_predict = APPLY.ridge_predict
cloud_rms = APPLY.cloud_rms


# ---------------------------------------------------------------- 官方评分的梯度

def chamfer_gradient(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """双向平方 Chamfer 对 pred 的解析梯度，两个方向各自按自身点数取均值。

    反向项（target -> pred 的最近邻）会把多个 target 点的贡献落到同一个 pred 点上，
    必须用 np.add.at 累加；直接下标赋值会只保留最后一个，梯度就错了。
    """
    pred_tree = cKDTree(pred)
    target_tree = cKDTree(target)
    pred_target_idx = target_tree.query(pred, k=1, workers=-1)[1]
    target_pred_idx = pred_tree.query(target, k=1, workers=-1)[1]
    gradient = (2.0 / len(pred)) * (pred - target[pred_target_idx])
    reverse = (2.0 / len(target)) * (pred[target_pred_idx] - target)
    np.add.at(gradient, target_pred_idx, reverse)
    return gradient


def normalized_cd_gradient(source: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """CD 项的梯度。

    官方 CD 先把 clean 归一化到单位球、再把 source 用同一 (center, scale) 带过去，
    所以梯度必须在归一化空间里求，再按链式法则除以 scale 换回原空间。
    """
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    centered = clean - center
    scale = float(np.sqrt(np.sum(centered * centered, axis=1)).max())
    if scale <= 1e-12:
        raise ValueError("clean 点云归一化尺度退化")
    clean_normalized = centered / scale
    source_normalized = (source - center) / scale
    return chamfer_gradient(source_normalized, clean_normalized) / scale


def closest_on_segment(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> np.ndarray:
    """点到线段的最近点，逐行向量化。退化线段（首尾重合）取起点。"""
    edge = end - start
    denominator = np.sum(edge * edge, axis=1)
    numerator = np.sum((points - start) * edge, axis=1)
    parameter = np.divide(
        numerator, denominator,
        out=np.zeros_like(numerator), where=denominator > 1e-30,
    )
    parameter = np.clip(parameter, 0.0, 1.0)
    return start + parameter[:, None] * edge


def p2s_gradient(
    source: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    center: np.ndarray,
    scale: float,
) -> np.ndarray:
    """P2S 项的梯度：2/N ×（点 − 网格上最近点）。

    网格先用 norm.json 的 (center, scale) 变到与点云同一坐标系，与评测脚本一致。

    退化三角面（面积为零的退化面，ShapeNet 里确实存在）上，点云工具返回的重心
    坐标可能是非有限值，插值出来的最近点就是 NaN。这时退回到「点到三条边的最近点
    取最近者」，并与工具返回的距离标量交叉校验，不一致就报错而不是静默用错的值。
    """
    import point_cloud_utils as pcu

    vertices_normalized = (vertices - center) / scale
    distances, face_ids, barycentric = pcu.closest_points_on_mesh(
        source.astype(np.float32),
        vertices_normalized.astype(np.float32),
        faces.astype(np.int32),
    )
    distances = np.asarray(distances, dtype=np.float64)
    face_ids = np.asarray(face_ids, dtype=np.int64)
    barycentric = np.asarray(barycentric, dtype=np.float64)
    triangles = vertices_normalized[faces[face_ids]]
    closest = np.einsum("ni,nij->nj", barycentric, triangles)

    bad = ~np.isfinite(closest).all(axis=1)
    if np.any(bad):
        bad_points = source[bad]
        bad_triangles = triangles[bad]
        candidates = np.stack([
            closest_on_segment(bad_points, bad_triangles[:, 0], bad_triangles[:, 1]),
            closest_on_segment(bad_points, bad_triangles[:, 1], bad_triangles[:, 2]),
            closest_on_segment(bad_points, bad_triangles[:, 2], bad_triangles[:, 0]),
        ], axis=1)
        squared = np.sum(np.square(candidates - bad_points[:, None, :]), axis=2)
        best = np.argmin(squared, axis=1)
        rows = np.arange(len(bad_points))
        fallback = candidates[rows, best]
        fallback_distances = np.sqrt(squared[rows, best])
        if np.max(np.abs(fallback_distances - distances[bad])) > 1e-5:
            raise RuntimeError("退化三角面兜底最近点与工具返回的距离不一致")
        closest[bad] = fallback

    gradient = (2.0 / len(source)) * (source - closest)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("P2S 梯度出现非有限值")
    return gradient


# -------------------------------------------------------------------- 拟合目标

def evaluate_scores(
    points: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
) -> dict[str, float]:
    """按官方口径给一个点云打分。noisy 那两项是评分的分母。"""
    cd_raw = float(EVALUATOR.chamfer_distance(points, clean, normalize=True))
    cd_noisy = float(EVALUATOR.chamfer_distance(noisy, clean, normalize=True))
    p2s_raw = float(EVALUATOR.point_to_surface_distance(
        points, vertices, faces, norm_center, norm_scale
    ))
    p2s_noisy = float(EVALUATOR.point_to_surface_distance(
        noisy, vertices, faces, norm_center, norm_scale
    ))
    cd_score = float(EVALUATOR.metric_to_score(cd_raw, cd_noisy))
    p2s_score = float(EVALUATOR.metric_to_score(p2s_raw, p2s_noisy))
    return {
        "cd_raw": cd_raw,
        "cd_noisy": cd_noisy,
        "p2s_raw": p2s_raw,
        "p2s_noisy": p2s_noisy,
        "cd_score": cd_score,
        "p2s_score": p2s_score,
        "final_score": 0.5 * (cd_score + p2s_score),
    }


def composite_ascent_target(
    source: np.ndarray,
    h: float,
    clean: np.ndarray,
    noisy: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    norm_center: np.ndarray,
    norm_scale: float,
) -> tuple[np.ndarray, dict[str, float], bool, bool]:
    """官方复合评分的上升方向，归一化到与应用侧同一剂量。

    返回 (无量纲逐点目标, 该形状的基准分数, CD 项是否计入, P2S 项是否计入)。
    """
    cd_gradient = normalized_cd_gradient(source, clean)
    p2s_grad = p2s_gradient(source, vertices, faces, norm_center, norm_scale)
    base = evaluate_scores(
        source, clean, noisy, vertices, faces, norm_center, norm_scale
    )

    # 评分被 clamp 到 0 或 100 的那一项，梯度对总分没有作用，置零而不是硬塞进去。
    cd_active = 0.0 < base["cd_raw"] < base["cd_noisy"]
    p2s_active = 0.0 < base["p2s_raw"] < base["p2s_noisy"]

    direction = np.zeros_like(source)
    if cd_active:
        direction -= cd_gradient / base["cd_noisy"]
    if p2s_active:
        direction -= p2s_grad / base["p2s_noisy"]

    rms = cloud_rms(direction)
    if rms <= 0.0 or not np.isfinite(direction).all():
        raise ValueError("复合评分上升方向非法（全零或含非有限值）")
    # 与 07_apply 的剂量口径对齐：位移 RMS = DOSE_OVER_H × h，除以 h 后无量纲。
    target = direction * (DOSE_OVER_H * h / rms) / h
    return target, base, cd_active, p2s_active


# -------------------------------------------------------------------- 岭回归

def deterministic_indices(shape_index: int) -> np.ndarray:
    """逐形状固定的抽点下标。种子按形状序号偏移，与清单顺序绑定。"""
    rng = np.random.Generator(np.random.PCG64(SAMPLE_SEED + shape_index))
    return np.sort(rng.choice(N_POINTS, size=TRAIN_POINTS, replace=False))


def basis_scale(bases: list[np.ndarray]) -> np.ndarray:
    """10 个基各自在拟合集上的 RMS。用来把量级差几十倍的基拉到同一尺度。"""
    stacked = np.concatenate(bases, axis=0)
    scale = np.sqrt(np.mean(np.square(stacked), axis=(0, 2)))
    if scale.shape != (10,) or np.any(scale <= 0.0) or not np.isfinite(scale).all():
        raise ValueError("基归一化尺度非法")
    return scale


def ridge_fit(
    bases: list[np.ndarray], targets: list[np.ndarray], lam: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """在归一化后的基上解岭回归。三个坐标分量共享同一组系数（各向同性）。"""
    scale = basis_scale(bases)
    x = np.concatenate([value / scale[None, :, None] for value in bases], axis=0)
    y = np.concatenate(targets, axis=0)
    # (N, 10, 3) -> (N×3, 10)：把 xyz 三个分量摊成独立样本，系数因此与坐标轴无关
    x2 = np.transpose(x, (0, 2, 1)).reshape(-1, 10)
    y2 = y.reshape(-1)
    gram = x2.T @ x2
    rhs = x2.T @ y2
    system = gram + lam * np.eye(10, dtype=np.float64)
    coefficients = np.linalg.solve(system, rhs)
    if not np.isfinite(coefficients).all():
        raise FloatingPointError("岭回归系数出现非有限值")
    return coefficients, scale, float(np.linalg.cond(system))


# ------------------------------------------------- 拟合用 source 的第二阶段推理

def infer_fit_source(
    datalist: Path, firstpass_cache: Path, stage2_ckpt: str, source_root: Path
) -> None:
    """用第二阶段权重在 52 个拟合形状上推理，产出拟合用的 source。

    走的是与最终提交完全相同的推理路径（run.py + 同一套 system/model 配置），
    只是输入清单换成 controller_fit_shapes.txt、输入目录换成第一阶段的训练集 cache。
    """
    rels = read_datalist(datalist)
    # run.py 的 CWD 是 starter_code/。cache 在代码包内就写相对路径，在包外就写绝对路径
    try:
        cache_ref = "../" + firstpass_cache.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        cache_ref = firstpass_cache.as_posix()
    try:
        list_ref = "./" + datalist.relative_to(STARTER_CODE).as_posix()
    except ValueError:
        raise SystemExit(
            f"[FAIL] 拟合清单必须放在 starter_code/ 下，run.py 才找得到: {datalist}"
        )
    run_tag = "repro_controller_fit_specialist"

    data_cfg_dir = STARTER_CODE / "configs" / "data" / VERSION
    task_cfg_dir = STARTER_CODE / "configs" / "task" / VERSION
    data_cfg_dir.mkdir(parents=True, exist_ok=True)
    task_cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_name = "_controller_fit_shapes_specialist"

    (data_cfg_dir / f"{cfg_name}.yaml").write_text(f"""CONFIG_VERSION: {VERSION}

# 第三阶段系数拟合用的第二阶段推理输入：第一阶段在 52 个拟合形状上的输出。
# 由 scripts/pipeline/07a_fit_direction_controller.py 生成。

predict_dataset:
  shuffle: False
  batch_size: 1
  num_workers: 0
  datapath:
    input_dataset_dir: {cache_ref}
    use_prob: False
    loader: npy
    data_name: denoised.npy
    ignore_check: True
    data_path:
      shapenet: [
        [{list_ref}, 1.0],
      ]
""", encoding="utf-8")

    (task_cfg_dir / f"{cfg_name}.yaml").write_text(f"""CONFIG_VERSION: {VERSION}
mode: predict
debug: False
load_ckpt: {stage2_ckpt}
run_tag: {run_tag}

components:
  data: {VERSION}/{cfg_name}
  transform: pdlts_light
  system: pdlts_light_predict
  model: pdlts_light

writer:
  __target__: pdlts_light
  save_dir: __overridden_by_system__
  save_name: denoised
""", encoding="utf-8")

    print(f"[拟合 source] 第二阶段推理 {len(rels)} 个形状，约 17 分钟")
    result = subprocess.run(
        [sys.executable, "run.py", "--task", f"configs/task/{VERSION}/{cfg_name}.yaml"],
        cwd=str(STARTER_CODE),
    )
    if result.returncode != 0:
        raise SystemExit(f"[FAIL] 第二阶段推理失败，run.py exit={result.returncode}")

    # run.py 每次调用生成带时间戳的新目录，取最新那个搬到固定路径，让下游可预测
    candidates = sorted(
        (REPO_ROOT / "outputs" / "predictions" / VERSION).glob(f"*{run_tag}*_predict"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise SystemExit(f"[FAIL] 没找到推理输出目录（run_tag={run_tag}）")
    produced = candidates[-1] / "pred"
    missing = [rel for rel in rels if not (produced / rel / "denoised.npy").is_file()]
    if missing:
        raise SystemExit(
            f"[FAIL] 推理输出缺 {len(missing)} 个样本：{missing[:5]}"
        )
    source_root.parent.mkdir(parents=True, exist_ok=True)
    if source_root.resolve() != produced.resolve():
        import shutil
        if source_root.exists():
            raise SystemExit(f"[FAIL] 目标目录已存在，拒绝覆盖: {source_root}")
        shutil.copytree(produced, source_root)
    print(f"[拟合 source] -> {source_root}")


def read_datalist(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8")
            if line.strip() and not line.startswith("#")]


def assert_no_leakage(samples: list[str]) -> None:
    """拟合集必须与官方测试集、本地验证集零重叠，不为零直接中止。

    拟合期会读真值与网格，所以这条必须是硬断言，不能只写在文档里。
    """
    fit = set(samples)
    for name in ("test.txt", "mock.txt", "mock_full.txt"):
        path = STARTER_CODE / "datalist" / name
        if not path.is_file():
            continue
        overlap = fit & set(read_datalist(path))
        if overlap:
            raise SystemExit(
                f"[FAIL] 拟合集与 {name} 有 {len(overlap)} 个重叠样本，中止: "
                f"{sorted(overlap)[:5]}"
            )
    print("  泄漏自检: 与 test / mock 清单交集为 0")


# -------------------------------------------------------------------- 主流程

def fit(
    samples: list[str], source_root: Path, train_root: Path, mesh_root: Path
) -> tuple[np.ndarray, np.ndarray, float, list[dict]]:
    """在拟合形状上逐个求目标、抽点，汇总后一次岭回归。"""
    all_bases: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    rows: list[dict] = []

    for shape_index, sample in enumerate(samples):
        source = np.load(source_root / sample / "denoised.npy").astype(np.float64)
        if source.shape != (N_POINTS, 3) or not np.isfinite(source).all():
            raise SystemExit(f"[FAIL] source 非法: {sample} shape={source.shape}")

        clean = np.load(train_root / sample / "clean.npy").astype(np.float64)
        noisy = np.load(train_root / sample / "noisy.npy").astype(np.float64)
        vertices, faces = EVALUATOR.load_mesh_vf(
            str(mesh_root / sample / "models/model_normalized.obj")
        )
        norm_center, norm_scale = EVALUATOR.load_norm(
            str(train_root / sample / "norm.json")
        )

        basis, h = multiscale_bases(source)
        target, base, cd_active, p2s_active = composite_ascent_target(
            source, h, clean, noisy, vertices, faces, norm_center, norm_scale
        )

        indices = deterministic_indices(shape_index)
        all_bases.append(basis[indices])
        all_targets.append(target[indices])
        rows.append({
            "sample": sample,
            "shape_index": shape_index,
            "median_nn": h,
            "cd_active": bool(cd_active),
            "p2s_active": bool(p2s_active),
            "base_cd_score": base["cd_score"],
            "base_p2s_score": base["p2s_score"],
            "base_final_score": base["final_score"],
        })
        print(
            f"  [{shape_index + 1}/{len(samples)}] {sample} "
            f"h={h:.6f} base_final={base['final_score']:.4f}",
            flush=True,
        )

    coefficients, scale, condition = ridge_fit(all_bases, all_targets, RIDGE_LAMBDA)
    return coefficients, scale, condition, rows


def compare_frozen(
    coefficients: np.ndarray, scale: np.ndarray, tolerance: float
) -> bool:
    """与 07_apply 里的冻结值逐项比较。"""
    frozen_coef = np.asarray(APPLY.COEFFICIENTS, dtype=np.float64)
    frozen_scale = np.asarray(APPLY.BASIS_SCALE, dtype=np.float64)
    coef_diff = np.abs(coefficients - frozen_coef)
    scale_diff = np.abs(scale - frozen_scale)

    print(f"\n[对拍] 与 07_apply_direction_controller.py 的冻结值比较"
          f"（阈值 {tolerance:.0e}）")
    print("  idx  拟合 coefficient        冻结 coefficient        绝对差")
    for i in range(10):
        print(f"  {i:>3}  {coefficients[i]:>22.17g}  {frozen_coef[i]:>22.17g}  "
              f"{coef_diff[i]:.3e}")
    print("  idx  拟合 basis_scale        冻结 basis_scale        绝对差")
    for i in range(10):
        print(f"  {i:>3}  {scale[i]:>22.17g}  {frozen_scale[i]:>22.17g}  "
              f"{scale_diff[i]:.3e}")

    max_coef = float(coef_diff.max())
    max_scale = float(scale_diff.max())
    print(f"\n  coefficients 最大绝对差: {max_coef:.3e}")
    print(f"  basis_scale  最大绝对差: {max_scale:.3e}")
    ok = max_coef <= tolerance and max_scale <= tolerance
    print(f"  结论: {'PASS 冻结值可复现' if ok else 'FAIL 与冻结值不一致'}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="第二阶段在 52 个拟合形状上的输出 pred/ 目录。"
                             "不存在且给了 --firstpass-cache 时会先推理生成")
    parser.add_argument("--datalist", default=DEFAULT_DATALIST,
                        help="拟合形状清单，顺序参与抽点种子，不要重排")
    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT,
                        help="训练集真值点云目录（clean/noisy/norm.json）")
    parser.add_argument("--mesh-root", default=DEFAULT_MESH_ROOT,
                        help="训练集网格目录，P2S 梯度需要")
    parser.add_argument("--firstpass-cache", default="",
                        help="第一阶段训练集 cache。source 缺失时用它推理生成 source")
    parser.add_argument("--stage2-ckpt", default=DEFAULT_STAGE2_CKPT,
                        help="第二阶段权重，仅在需要推理生成 source 时使用。"
                             "starter_code/ 相对路径（run.py 的 CWD 是 starter_code/）")
    parser.add_argument("--out", default=DEFAULT_OUT, help="拟合结果输出目录")
    parser.add_argument("--verify-frozen", action="store_true",
                        help="与 07_apply 里的冻结值对拍，不一致则以非零码退出")
    parser.add_argument("--tolerance", type=float, default=1e-12,
                        help="对拍阈值。默认 1e-12")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = resolve(args.source)
    datalist = resolve(args.datalist)
    train_root = resolve(args.train_root)
    mesh_root = resolve(args.mesh_root)
    out_root = resolve(args.out)

    samples = read_datalist(datalist)
    if len(samples) != 52:
        raise SystemExit(f"[FAIL] 拟合清单应有 52 个形状，实际 {len(samples)}")

    if not source_root.is_dir():
        if not args.firstpass_cache:
            raise SystemExit(
                f"[FAIL] source 目录不存在: {source_root}\n"
                f"        要么给 --source 指向已有的第二阶段输出，"
                f"要么给 --firstpass-cache 让本脚本先推理生成"
            )
        infer_fit_source(
            datalist, resolve(args.firstpass_cache), args.stage2_ckpt, source_root
        )

    missing = [s for s in samples if not (source_root / s / "denoised.npy").is_file()]
    if missing:
        raise SystemExit(
            f"[FAIL] source 缺 {len(missing)} 个形状: {missing[:5]}"
        )
    for root, label in ((train_root, "训练集真值"), (mesh_root, "训练集网格")):
        if not root.is_dir():
            raise SystemExit(f"[FAIL] {label}目录不存在: {root}")

    print(f"[方向场系数拟合] script_version={SCRIPT_VERSION}")
    print(f"  source:     {source_root}")
    print(f"  真值:       {train_root}")
    print(f"  网格:       {mesh_root}")
    print(f"  形状数:     {len(samples)}")
    print(f"  抽点:       每形状 {TRAIN_POINTS} 点，种子 {SAMPLE_SEED} + 形状序号")
    print(f"  岭回归:     lambda={RIDGE_LAMBDA}")
    print(f"  剂量:       RMS = {DOSE_OVER_H} × h")
    assert_no_leakage(samples)
    print()

    t0 = time.time()
    coefficients, scale, condition, rows = fit(
        samples, source_root, train_root, mesh_root
    )
    elapsed = time.time() - t0
    print(f"\n[拟合完成] {elapsed:.1f}s，岭系统条件数 {condition:.3e}")

    ok = True
    if args.verify_frozen:
        ok = compare_frozen(coefficients, scale, args.tolerance)

    model = {
        "script": "scripts/pipeline/07a_fit_direction_controller.py",
        "script_version": SCRIPT_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(source_root),
        "train_root": str(train_root),
        "mesh_root": str(mesh_root),
        "datalist": str(datalist),
        "n_fit_shapes": len(samples),
        "k_values": list(K_VALUES),
        "dose_over_h": DOSE_OVER_H,
        "ridge_lambda": RIDGE_LAMBDA,
        "train_points_per_shape": TRAIN_POINTS,
        "sample_seed": SAMPLE_SEED,
        "coefficients": coefficients.tolist(),
        "basis_scale": scale.tolist(),
        "condition_number": condition,
        "wall_sec": round(elapsed, 2),
        "reads_train_ground_truth": True,
        "reads_train_mesh": True,
        "reads_test_set": False,
        "rows": rows,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "controller_model.json").open("w", encoding="utf-8") as f:
        json.dump(model, f, indent=2, ensure_ascii=False)
    print(f"  模型 -> {out_root / 'controller_model.json'}")

    print("\n可直接粘回 07_apply_direction_controller.py 的常量：")
    print("COEFFICIENTS = (")
    for value in coefficients:
        print(f"    {value!r},")
    print(")")
    print("BASIS_SCALE = (")
    for value in scale:
        print(f"    {value!r},")
    print(")")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
