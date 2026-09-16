#!/usr/bin/env bash
set -euo pipefail

# 从赛事原始 B 榜训练 mesh 完成训练对、base（基础模型）、first-pass cache（第一遍缓存）
# 和第二遍 specialist（专训模型）。

usage() {
    cat <<'EOF'
用法：
  bash scripts/experiments/b_final/train_b_final_pipeline.sh

说明：
  从 dataset/train_b 生成训练对，训练第一遍 base，生成第二遍缓存，
  再训练 specialist。完整流程需要 GPU、官方 B 榜训练 mesh 和较长运行时间。
EOF
}

case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUNS_ROOT="$CODE_ROOT/outputs/runs/b_final"
PRED_ROOT="$CODE_ROOT/outputs/predictions/b_final"
STAMP="$(date +%Y%m%d_%H%M%S)"
CACHE_RUN_ID="${STAMP}_b_final_firstpass_train2k_merged_predict"
SPECIALIST_RUN_ID="${STAMP}_b_final_specialist_train"

cd "$CODE_ROOT"

echo "[1/4] generate deterministic B training pairs and healthy datalist"
python scripts/experiments/b_final/generate_b_training_pairs.py \
    --dataset-dir dataset/train_b \
    --datalist starter_code/datalist/train_b.txt \
    --output-datalist starter_code/datalist/train_b_generated.txt \
    --expected-success 19528 \
    --num-workers 8

echo "[2/4] train base epoch 0..149"
bash scripts/experiments/b_final/train_base_pipeline.sh

BASE_RUN="$(find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name '*b_final_base_resume_ep100_149*' -printf '%T@ %p\n' \
    | sort -nr | head -1 | cut -d' ' -f2-)"
BASE_CKPT="$BASE_RUN/checkpoints/pdlts_light_149.pkl"
if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: epoch 149 checkpoint not found: $BASE_CKPT"
    exit 1
fi

echo "[3/4] generate first-pass cache for the fixed 2000-shape specialist set"
python scripts/experiments/b_final/generate_firstpass_cache.py \
    --datalist starter_code/datalist/b_final_specialist_train2k.txt \
    --train-root dataset/train_b \
    --base-ckpt "$BASE_CKPT" \
    --chunk-size 150 \
    --run-tag-prefix b_final_firstpass_cache \
    --merged-run-id "$CACHE_RUN_ID"

CACHE_DIR="$PRED_ROOT/$CACHE_RUN_ID/pred"
N_CACHE="$(find "$CACHE_DIR" -name denoised.npy | wc -l)"
if [ "$N_CACHE" -ne 2000 ]; then
    echo "ERROR: expected 2000 cache outputs, found $N_CACHE"
    exit 1
fi

echo "[4/4] train second-pass specialist"
python scripts/experiments/b_final/train_deep_specialist.py \
    --stage specialist \
    --load-ckpt "$BASE_CKPT" \
    --firstpass-cache "$CACHE_DIR" \
    --clean-dir dataset/train_b \
    --train-datalist starter_code/datalist/b_final_specialist_train2k.txt \
    --run-id "$SPECIALIST_RUN_ID"

SPECIALIST_CKPT="$RUNS_ROOT/$SPECIALIST_RUN_ID/b_specialist_final.pkl"
if [ ! -f "$SPECIALIST_CKPT" ]; then
    echo "ERROR: specialist checkpoint not found: $SPECIALIST_CKPT"
    exit 1
fi

echo "BASE_CKPT=$BASE_CKPT"
echo "SPECIALIST_CKPT=$SPECIALIST_CKPT"
echo "[PASS] complete B-board training pipeline finished"
