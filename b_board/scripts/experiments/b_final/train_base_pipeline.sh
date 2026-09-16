#!/usr/bin/env bash
set -euo pipefail

# 训练 B 榜 base（基础模型）：epoch 0 到 99 从零训练，epoch 100 到 149 strict resume（严格续训）。

usage() {
    cat <<'EOF'
用法：
  bash scripts/experiments/b_final/train_base_pipeline.sh

说明：
  训练第一遍 base：先从零训练 100 个 epoch，再严格续训到 epoch 149。
  需要先生成 starter_code/datalist/train_b_generated.txt。
EOF
}

case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STARTER_ROOT="$CODE_ROOT/starter_code"
RUNS_ROOT="$CODE_ROOT/outputs/runs/b_final"
RESUME_TEMPLATE="$STARTER_ROOT/configs/task/b_final/train_base_resume_ep100_149.yaml"
RUNTIME_TASK="$STARTER_ROOT/configs/task/b_final/_runtime_train_base_resume.yaml"

latest_run() {
    local pattern="$1"
    find "$RUNS_ROOT" -mindepth 1 -maxdepth 1 -type d -name "$pattern" \
        -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-
}

cleanup() {
    rm -f "$RUNTIME_TASK"
}
trap cleanup EXIT

if [ ! -f "$STARTER_ROOT/datalist/train_b_generated.txt" ]; then
    echo "ERROR: missing starter_code/datalist/train_b_generated.txt"
    echo "Run scripts/experiments/b_final/generate_b_training_pairs.py first."
    exit 1
fi

N_TRAIN="$(grep -cve '^[[:space:]]*$' "$STARTER_ROOT/datalist/train_b_generated.txt")"
if [ "$N_TRAIN" -ne 19528 ]; then
    echo "ERROR: expected 19528 healthy training samples, found $N_TRAIN"
    exit 1
fi

mkdir -p "$RUNS_ROOT"
cd "$STARTER_ROOT"

echo "[1/2] scratch training epoch 0..99"
python run.py --task b_final/train_base_scratch_ep100

SCRATCH_RUN="$(latest_run '*b_final_base_scratch_ep100*')"
if [ -z "$SCRATCH_RUN" ]; then
    echo "ERROR: scratch run directory not found"
    exit 1
fi
RESUME_STATE="$SCRATCH_RUN/checkpoints/pdlts_light_99.train.pkl"
if [ ! -f "$RESUME_STATE" ]; then
    echo "ERROR: strict-resume state not found: $RESUME_STATE"
    exit 1
fi

RESUME_REL="$(python - "$STARTER_ROOT" "$RESUME_STATE" <<'PY'
import os
import sys
print(os.path.relpath(sys.argv[2], sys.argv[1]).replace(os.sep, "/"))
PY
)"

python - "$RESUME_TEMPLATE" "$RUNTIME_TASK" "$RESUME_REL" <<'PY'
import sys
from omegaconf import OmegaConf

template, runtime, resume_state = sys.argv[1:]
config = OmegaConf.load(template)
config.resume_state = resume_state
OmegaConf.save(config=config, f=runtime)
PY

echo "[2/2] strict resume epoch 100..149"
python run.py --task b_final/_runtime_train_base_resume

RESUME_RUN="$(latest_run '*b_final_base_resume_ep100_149*')"
if [ -z "$RESUME_RUN" ]; then
    echo "ERROR: strict-resume run directory not found"
    exit 1
fi
BASE_CKPT="$RESUME_RUN/checkpoints/pdlts_light_149.pkl"
if [ ! -f "$BASE_CKPT" ]; then
    echo "ERROR: final epoch 149 checkpoint not found: $BASE_CKPT"
    exit 1
fi

echo "BASE_CKPT=$BASE_CKPT"
echo "[PASS] B-board base training completed"
