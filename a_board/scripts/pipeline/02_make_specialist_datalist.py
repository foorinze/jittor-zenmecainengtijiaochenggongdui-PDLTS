#!/usr/bin/env python3
"""生成第二阶段训练子集的样本清单，按类别配额抽样并断言无泄漏。

纯 CPU，不需要 GPU。

为什么需要这一步：第二阶段只用训练集的一个子集（2000 个样本），而随手抽
子集会引入类别偏斜。官方训练集里 table 与 airplane 两类合计占 63%，早期
按其他方式抽出的子集把这两类稀释到 17%，子集的类别构成与训练集总体差得
很远。本脚本改为按总体各类占比分配配额，让子集的类别分布还原训练集总体。

抽样算法：每类配额 = round(target_n * 该类样本数 / 总体样本数)，用固定
seed 的无放回抽样。若某类样本数少于配额则取全量，不重复采样。配额总数可能
因四舍五入与 target_n 差几个，不强制补齐，实际值记录在清单里。

泄漏处理：训练集总体清单本身与本地验证集有交集（本地验证集是从训练集
构造出来的，两者天然重叠）。因此本脚本先从总体池剔除与本地验证集、
validate、官方 test 清单的全部交集，再在干净池子里抽配额，最后再显式断言
一次交集为 0，任一不为 0 就中止不写盘。

用法：
    python scripts/pipeline/02_make_specialist_datalist.py \\
        --target-n 2000 --seed 42 \\
        --out starter_code/datalist/specialist_train2000.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

VERSION = "a_final"
SCRIPT_VERSION = "public"

DEFAULT_MOTHER_DATALIST = "starter_code/datalist/train_full15k.txt"
LEAK_CHECK_DATALISTS = {
    "mock_full": "starter_code/datalist/mock_full.txt",
    "validate": "starter_code/datalist/validate.txt",
    "test": "starter_code/datalist/test.txt",
}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def read_datalist(path: Path) -> list[str]:
    return [line.strip() for line in path.open("r", encoding="utf-8")
            if line.strip() and not line.startswith("#")]


def class_of(rel: str) -> str:
    return rel.split("/")[1]


def quota_sample(
    mother_rels: list[str], target_n: int, seed: int
) -> tuple[list[str], list[dict]]:
    """按类别占比配额无放回抽样，返回选中的 rels（未打乱输出顺序，按训练集总体出现顺序）。"""
    by_class: dict[str, list[str]] = {}
    for rel in mother_rels:
        by_class.setdefault(class_of(rel), []).append(rel)

    n_mother = len(mother_rels)
    rng = random.Random(seed)
    selected: list[str] = []
    quota_report = []
    for cls, rels in sorted(by_class.items()):
        target_quota = round(target_n * len(rels) / n_mother)
        take = min(target_quota, len(rels))
        chosen = rng.sample(rels, take) if take > 0 else []
        selected.extend(chosen)
        quota_report.append({
            "class": cls,
            "mother_count": len(rels),
            "mother_ratio": len(rels) / n_mother,
            "target_quota": target_quota,
            "actual_quota": take,
        })
    return selected, quota_report


def leak_check(selected: set[str]) -> dict[str, int]:
    result = {}
    for name, rel_path in LEAK_CHECK_DATALISTS.items():
        other = set(read_datalist(resolve(rel_path)))
        overlap = selected & other
        result[name] = len(overlap)
    return result


def leaked_rels(mother_rels: list[str]) -> set[str]:
    """训练集总体清单与本地验证集/validate/官方 test 清单的交集。

    本地验证集是从 dataset/train 构造的，训练集总体清单覆盖了 train 的绝大
    部分，所以两者天然有交集：实测总体清单与本地 200 样本验证集重叠 178 个。
    这不是本脚本引入的问题，是这个池子本身的特征。

    注意比较时必须先 strip 掉行尾符。总体清单是 CRLF 换行，逐行裸比较会把
    "abc\\r" 和 "abc" 当成两个不同样本，从而误判为零重叠。read_datalist()
    已统一 strip。

    处理方式：先剔除全部交集，再在剩余的干净池子里抽配额，保证产出的清单
    天生无泄漏。
    """
    leak_set: set[str] = set()
    for rel_path in LEAK_CHECK_DATALISTS.values():
        leak_set |= set(read_datalist(resolve(rel_path)))
    mother_set = set(mother_rels)
    return leak_set & mother_set


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mother-datalist", default=DEFAULT_MOTHER_DATALIST)
    parser.add_argument("--target-n", type=int, required=True,
                         help="目标规模（6000/4000/2000，按 timing gate 结果选择）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True,
                         help="starter_code/datalist/ 下的输出路径")
    args = parser.parse_args()

    mother_path = resolve(args.mother_datalist)
    mother_rels_raw = read_datalist(mother_path)
    print(f"[INFO] script_version={SCRIPT_VERSION}")
    print(f"[INFO] mother_datalist={mother_path} (n_raw={len(mother_rels_raw)})")
    print(f"[INFO] target_n={args.target_n} seed={args.seed}")

    leaked = leaked_rels(mother_rels_raw)
    print(f"\n[泄漏剔除] 总体清单与本地验证集/validate/官方 test 的交集 "
          f"= {len(leaked)}，先剔除再抽样")
    mother_rels = [rel for rel in mother_rels_raw if rel not in leaked]
    print(f"[泄漏剔除] 剔除后训练集总体池 n={len(mother_rels)} "
          f"(raw {len(mother_rels_raw)} - leaked {len(leaked)})")

    selected, quota_report = quota_sample(mother_rels, args.target_n, args.seed)
    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise SystemExit(
            f"[FAIL] 抽样产生重复 rel（{len(selected)} 抽取值 vs "
            f"{len(selected_set)} 去重值），采样逻辑有 bug，停止写盘。"
        )

    print(f"\n[配额表] (按训练集总体类别占比配额)")
    print(f"{'class':12s} {'mother_n':>10s} {'ratio':>8s} {'target':>8s} {'actual':>8s}")
    for row in quota_report:
        print(f"{row['class']:12s} {row['mother_count']:10d} "
              f"{row['mother_ratio']*100:7.3f}% {row['target_quota']:8d} {row['actual_quota']:8d}")
    print(f"\n[TOTAL] target_n={args.target_n} actual_n={len(selected)} "
          f"(diff={len(selected) - args.target_n}, 四舍五入累积偏差，不强制补齐)")

    print("\n[泄漏断言] 与 mock/validate/test 的交集（必须全部为 0）")
    leak = leak_check(selected_set)
    for name, count in leak.items():
        print(f"  {name}: overlap={count}")
    if any(v > 0 for v in leak.values()):
        raise SystemExit(
            f"[FAIL] 泄漏断言未通过：{leak}。停止写盘，需要用户介入排查训练集总体"
            f"datalist 本身是否与评测集有交集。"
        )
    print("[泄漏断言] PASS（全部交集为 0）")

    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rel in selected:
            f.write(rel + "\n")

    content_hash = hashlib.sha256("\n".join(sorted(selected)).encode("utf-8")).hexdigest()

    manifest_dir = (REPO_ROOT / "outputs" / "diagnostics" / VERSION
                     / f"{time.strftime('%Y%m%d_%H%M%S')}_a_final_make_specialist_datalist")
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script": "scripts/pipeline/02_make_specialist_datalist.py",
        "script_version": SCRIPT_VERSION,
        "mother_datalist": str(mother_path),
        "n_mother_raw": len(mother_rels_raw),
        "n_mother_leaked_purged": len(leaked),
        "n_mother_clean": len(mother_rels),
        "target_n": args.target_n,
        "actual_n": len(selected),
        "seed": args.seed,
        "out_datalist": str(out_path),
        "content_sha256": content_hash,
        "quota_report": quota_report,
        "leak_check": leak,
        "leak_check_datalists": LEAK_CHECK_DATALISTS,
        "note": "本步作用是还原训练集总体的类别构成、修掉子集抽样偏斜；"
                "不涉及官方测试集的任何信息。",
    }
    with (manifest_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[DONE] datalist -> {out_path}")
    print(f"[DONE] manifest -> {manifest_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
