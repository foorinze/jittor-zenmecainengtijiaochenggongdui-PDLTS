#!/usr/bin/env python3
"""用第一阶段模型跑出训练子集的输出，作为第二阶段的训练输入。

第二阶段学的是「第一阶段输出 -> 真值」的残余映射，所以必须先把第一阶段
在训练子集上的输出落盘。2000 个 50k 点整云的推理不可能放在训练循环里现算，
必须离线跑一遍缓存下来。

分 chunk 跑：每个 chunk 单独起一个 run.py 子进程，跑完进程退出，显存与
JIT 编译缓存自然释放。单进程连续推理 2000 个整云会内存碎片化并最终失败，
这不是并行优化而是稳定性要求。

推理口径与最终提交完全一致：seed_k=12。alpha 用 80 而不是默认的 30，
alpha 只控制 patch 分几批送进网络（batch_size = ceil(K / alpha)），
不改变任何数值结果，只是把显存峰值压低，让长时间批量推理不会中途 OOM。

跑完后所有 chunk 的输出会合并到一个目录，因为下游训练脚本需要「一个目录
含全部样本」的形式。

用法：
    python scripts/pipeline/03_generate_firstpass_cache.py \\
        --datalist starter_code/datalist/specialist_train2000.txt \\
        --chunk-size 150
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
STARTER_CODE = REPO_ROOT / "starter_code"

VERSION = "a_final"
SCRIPT_VERSION = "public"

DEFAULT_TRAIN_ROOT = "dataset/a_final_train_full15k"
DEFAULT_BASE_CKPT = (
    "../outputs/runs/a_final/repro_base_train/checkpoints/pdlts_light_99.pkl"
)
DEFAULT_MODEL_CONFIG = "pdlts_light_lowmem"


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_datalist(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8")
            if line.strip() and not line.startswith("#")]


def write_chunk_files(rels: list[str], chunk_dir: Path, chunk_name: str) -> Path:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"{chunk_name}.txt"
    with chunk_path.open("w", encoding="utf-8") as f:
        for rel in rels:
            f.write(rel + "\n")
    return chunk_path


def write_data_config(config_path: Path, train_root: str, chunk_datalist_starter_rel: str) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"""CONFIG_VERSION: a_final

predict_dataset:
  shuffle: False
  batch_size: 1
  num_workers: 0
  datapath:
    input_dataset_dir: ../{train_root}
    use_prob: False
    loader: npy
    data_name: noisy.npy
    ignore_check: True
    data_path:
      shapenet: [
        [{chunk_datalist_starter_rel}, 1.0],
      ]
""", encoding="utf-8")


def write_task_config(
    config_path: Path, data_config: str, model_config: str, base_ckpt: str, run_tag: str
) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"""CONFIG_VERSION: a_final
mode: predict
debug: False
load_ckpt: {base_ckpt}
run_tag: {run_tag}

components:
  data: {data_config}
  transform: pdlts_light
  system: pdlts_light_predict
  model: {model_config}

writer:
  __target__: pdlts_light
  save_dir: __overridden_by_system__
  save_name: denoised
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--base-ckpt", default=DEFAULT_BASE_CKPT,
                         help="starter_code/ 相对路径（run.py 的 CWD 是 starter_code/）")
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--chunk-size", type=int, default=150,
                         help="每个 chunk 的样本数。150 是实测稳定值")
    parser.add_argument("--run-tag-prefix", default="a_final_firstpass_cache")
    parser.add_argument("--merged-run-id", default="",
                         help="合并后目录名。留空则用时间戳，传入固定值可让路径可预测")
    parser.add_argument("--dry-run", action="store_true",
                         help="只生成 chunk 文件与 configs，不实际跑 run.py")
    parser.add_argument("--start-chunk", type=int, default=0,
                         help="从第几个 chunk（0-indexed）开始跑，用于崩溃后续跑，"
                              "跳过已确认完整的前置 chunk（不重复计入 timing/耗时统计）")
    args = parser.parse_args()

    datalist_path = resolve(args.datalist)
    rels = read_datalist(datalist_path)
    print(f"[INFO] script_version={SCRIPT_VERSION}")
    print(f"[INFO] datalist={datalist_path} (n={len(rels)})")
    print(f"[INFO] train_root={args.train_root}")
    print(f"[INFO] base_ckpt={args.base_ckpt}")
    print(f"[INFO] chunk_size={args.chunk_size}")

    n_chunks = (len(rels) + args.chunk_size - 1) // args.chunk_size
    print(f"[INFO] n_chunks={n_chunks}")

    datalist_tag = datalist_path.stem
    chunk_dir = STARTER_CODE / "datalist" / f"_firstpass_chunks_{datalist_tag}"
    data_config_dir = STARTER_CODE / "configs" / "data" / VERSION
    task_config_dir = STARTER_CODE / "configs" / "task" / VERSION

    if args.start_chunk > 0:
        print(f"[INFO] start_chunk={args.start_chunk}（跳过前 {args.start_chunk} 个已确认完整的 chunk，"
              f"续跑模式，不重新生成/重跑它们的 chunk 文件与 predict）")

    t0_all = time.time()
    for ci in range(args.start_chunk, n_chunks):
        chunk_rels = rels[ci * args.chunk_size: (ci + 1) * args.chunk_size]
        chunk_name = f"_firstpass_{datalist_tag}_chunk_{ci:02d}"
        chunk_path = write_chunk_files(chunk_rels, chunk_dir, chunk_name)
        chunk_rel_from_starter = f"./{chunk_path.relative_to(STARTER_CODE).as_posix()}"

        data_cfg_name = f"_firstpass_cache_chunk_{datalist_tag}"
        task_cfg_name = f"_firstpass_cache_chunk_{datalist_tag}"
        write_data_config(
            data_config_dir / f"{data_cfg_name}.yaml", args.train_root, chunk_rel_from_starter)
        run_tag = f"{args.run_tag_prefix}_{datalist_tag}_chunk{ci:02d}"
        write_task_config(
            task_config_dir / f"{task_cfg_name}.yaml",
            f"{VERSION}/{data_cfg_name}", args.model_config, args.base_ckpt, run_tag)

        print(f"\n=== chunk {ci+1}/{n_chunks}: {chunk_name} ({len(chunk_rels)} samples) ===")
        if args.dry_run:
            print(f"  [DRY-RUN] would run: python run.py --task configs/task/{VERSION}/{task_cfg_name}.yaml")
            continue

        t0 = time.time()
        result = subprocess.run(
            [sys.executable, "run.py", "--task", f"configs/task/{VERSION}/{task_cfg_name}.yaml"],
            cwd=str(STARTER_CODE),
        )
        elapsed = time.time() - t0
        print(f"  chunk {ci+1} done in {elapsed:.1f}s (exit={result.returncode})")
        if result.returncode != 0:
            raise SystemExit(f"[FAIL] chunk {ci} (run_tag={run_tag}) run.py exit={result.returncode}")

    total_elapsed = time.time() - t0_all
    print(f"\n[DONE] all {n_chunks} chunks in {total_elapsed:.1f}s ({total_elapsed/3600:.2f}h)")

    if args.dry_run:
        print("[DRY-RUN] skip merge step")
        return 0

    # 每个 chunk 各自落在一个带时间戳的独立输出目录（run.py 每次调用生成新
    # run_id）。下游训练脚本要的是"单个目录含全部样本"的形式，所以这里把所有
    # chunk 的 pred/ 合并到一个新目录。
    import shutil

    predictions_root = REPO_ROOT / "outputs" / "predictions" / VERSION
    candidates = sorted(
        predictions_root.glob(f"*{args.run_tag_prefix}_{datalist_tag}_chunk*_predict"),
        key=lambda p: p.stat().st_mtime,
    )
    if len(candidates) < n_chunks:
        raise SystemExit(
            f"[FAIL] 找到 {len(candidates)} 个 chunk 预测目录，预期 {n_chunks} 个，"
            f"合并前中止（避免合并不完整的 cache）。"
        )
    # 取最近一批（时间戳最新的 n_chunks 个），防止历史同名残留混入
    candidates = candidates[-n_chunks:]

    merged_name = args.merged_run_id or (
        time.strftime("%Y%m%d_%H%M%S")
        + f"_{args.run_tag_prefix}_{datalist_tag}_merged_predict"
    )
    merged_dir = predictions_root / merged_name
    merged_pred_dir = merged_dir / "pred"
    merged_pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[MERGE] merging {len(candidates)} chunk dirs -> {merged_pred_dir}")
    for cand in candidates:
        src_pred = cand / "pred"
        if not src_pred.is_dir():
            raise SystemExit(f"[FAIL] chunk 目录缺 pred/: {cand}")
        for npy_path in src_pred.rglob("denoised.npy"):
            rel = npy_path.relative_to(src_pred)
            dst_path = merged_pred_dir / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(npy_path, dst_path)

    n_merged = sum(1 for _ in merged_pred_dir.rglob("denoised.npy"))
    missing_after_merge = [rel for rel in rels
                            if not (merged_pred_dir / rel / "denoised.npy").exists()]
    print(f"[MERGE] merged {n_merged} denoised.npy files")
    if missing_after_merge:
        raise SystemExit(
            f"[FAIL] 合并后仍缺 {len(missing_after_merge)} 个样本："
            f"{missing_after_merge[:5]}"
        )
    print(f"[MERGE] 完整性核查 PASS：{len(rels)}/{len(rels)} 样本齐")
    print(f"\n[DONE] merged cache -> {merged_pred_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
