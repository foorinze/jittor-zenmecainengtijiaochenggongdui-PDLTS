#!/usr/bin/env bash
set -euo pipefail

# Public entry for reproducing the final B-board submission.

CALLER_DIR="$PWD"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STARTER_ROOT="$SCRIPT_DIR/starter_code"
PRED_ROOT="$SCRIPT_DIR/outputs/predictions/b_final"
PYTHON_BIN="${PYTHON_BIN:-python}"
JITTOR_COMPILER_THREADS="${JITTOR_COMPILER_THREADS:-1}"
JITTOR_TASK_RUNNER="$SCRIPT_DIR/scripts/shared/run_jittor_task.py"
BASE_CKPT_ARG="$SCRIPT_DIR/checkpoints/base_ep149.pkl"
SPECIALIST_CKPT_ARG="$SCRIPT_DIR/checkpoints/specialist_final.pkl"

if ! [[ "$JITTOR_COMPILER_THREADS" =~ ^[0-9]+$ ]]; then
    echo "ERROR: JITTOR_COMPILER_THREADS 必须是非负整数" >&2
    exit 2
fi

usage() {
    cat <<'EOF'
用法：
  bash b_board/reproduce_b_final.sh [选项]

选项：
  --base-ckpt PATH        第一遍 base（基础模型）权重
  --specialist-ckpt PATH  第二遍 specialist（专训模型）权重
  -h, --help              显示帮助

相对路径按调用脚本时的工作目录解析。省略两个权重参数时，使用提交包随附权重。
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --base-ckpt)
            [ "$#" -ge 2 ] || { echo "ERROR: --base-ckpt requires PATH"; exit 2; }
            BASE_CKPT_ARG="$2"
            shift 2
            ;;
        --specialist-ckpt)
            [ "$#" -ge 2 ] || { echo "ERROR: --specialist-ckpt requires PATH"; exit 2; }
            SPECIALIST_CKPT_ARG="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            usage
            exit 2
            ;;
    esac
done

resolve_input_file() {
    local value="$1"
    local candidate
    if [[ "$value" = /* ]]; then
        candidate="$value"
    else
        candidate="$CALLER_DIR/$value"
    fi
    if [ ! -f "$candidate" ]; then
        echo "ERROR: checkpoint not found: $candidate" >&2
        return 1
    fi
    readlink -f "$candidate"
}

BASE_CKPT="$(resolve_input_file "$BASE_CKPT_ARG")"
SPECIALIST_CKPT="$(resolve_input_file "$SPECIALIST_CKPT_ARG")"

PASS1_TASK="$STARTER_ROOT/configs/task/b_final/_runtime_reproduce_pass1.yaml"
PASS2_TASK="$STARTER_ROOT/configs/task/b_final/_runtime_reproduce_pass2.yaml"
PASS2_DATA="$STARTER_ROOT/configs/data/b_final/_runtime_reproduce_cascade.yaml"

cleanup() {
    rm -f "$PASS1_TASK" "$PASS2_TASK" "$PASS2_DATA"
}
trap cleanup EXIT

if [ ! -f "$STARTER_ROOT/run.py" ]; then
    echo "ERROR: run.py not found: $STARTER_ROOT/run.py"
    exit 1
fi

if [ ! -d "$SCRIPT_DIR/dataset/test_noisy_b/shapenet/00000000" ]; then
    echo "ERROR: B 榜测试数据未找到。"
    echo "请放到：$SCRIPT_DIR/dataset/test_noisy_b/shapenet/00000000/"
    echo "提交包按比赛规则不包含数据集。"
    exit 1
fi

N_SAMPLES="$(find "$SCRIPT_DIR/dataset/test_noisy_b/shapenet/00000000" \
    -mindepth 1 -maxdepth 1 -type d | wc -l)"
if [ "$N_SAMPLES" -ne 200 ]; then
    echo "ERROR: B 榜测试集应有 200 个样本，实际为 $N_SAMPLES"
    exit 1
fi

cat > "$PASS1_TASK" <<EOF
CONFIG_VERSION: b_final
mode: predict
debug: False
load_ckpt: "$BASE_CKPT"
components:
  data: b_final/official_b_test_noisy
  transform: _shared/pdlts_light
  system: b_final/pass1_base_official_b_predict
  model: b_final/final_light_lowmem
writer:
  __target__: pdlts_light
  save_dir: __overridden_by_system__
  save_name: denoised
EOF

mkdir -p "$PRED_ROOT"
cd "$STARTER_ROOT"

echo "[1/5] 权重与测试数据检查通过"
echo "  base: $BASE_CKPT"
echo "  specialist: $SPECIALIST_CKPT"
echo "  test samples: $N_SAMPLES"
echo "  Jittor compiler threads: $JITTOR_COMPILER_THREADS"

echo "[2/5] 第一遍：base checkpoint，seed_k=12/24"
PASS1_BEFORE="$(mktemp)"
find "$PRED_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$PASS1_BEFORE"
JITTOR_COMPILER_THREADS="$JITTOR_COMPILER_THREADS" "$PYTHON_BIN" \
    "$JITTOR_TASK_RUNNER" --task b_final/_runtime_reproduce_pass1
PASS1_RUN="$(comm -13 "$PASS1_BEFORE" \
    <(find "$PRED_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
    | grep 'b_final_pass1_base_official_b' | tail -1 || true)"
rm -f "$PASS1_BEFORE"
if [ -z "$PASS1_RUN" ]; then
    echo "ERROR: 未找到第一遍新生成的预测目录"
    exit 1
fi

PASS1_DIR="$PRED_ROOT/$PASS1_RUN/pred"
PASS1_MANIFEST="$PRED_ROOT/$PASS1_RUN/manifest.json"
N_PRED1="$(find "$PASS1_DIR/shapenet/00000000" -name denoised.npy 2>/dev/null | wc -l)"
if [ "$N_PRED1" -ne 200 ] || [ ! -f "$PASS1_MANIFEST" ]; then
    echo "ERROR: 第一遍输出不完整：predictions=$N_PRED1 manifest=$PASS1_MANIFEST"
    exit 1
fi
VERDICT1="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["verdict"])' "$PASS1_MANIFEST")"
if [ "$VERDICT1" != "green" ]; then
    echo "ERROR: 第一遍 manifest verdict=$VERDICT1"
    exit 1
fi

cat > "$PASS2_DATA" <<EOF
CONFIG_VERSION: b_final
predict_dataset:
  shuffle: False
  batch_size: 1
  num_workers: 0
  datapath:
    input_dataset_dir: "$PASS1_DIR"
    use_prob: False
    loader: npy
    data_name: denoised.npy
    ignore_check: True
    data_path:
      shapenet:
        - [./datalist/test_b.txt, 1.0]
EOF

cat > "$PASS2_TASK" <<EOF
CONFIG_VERSION: b_final
mode: predict
debug: False
load_ckpt: "$SPECIALIST_CKPT"
components:
  data: b_final/_runtime_reproduce_cascade
  transform: _shared/pdlts_light
  system: b_final/pass2_specialist_cascade_official_b_predict
  model: b_final/final_light_lowmem
writer:
  __target__: pdlts_light
  save_dir: __overridden_by_system__
  save_name: denoised
EOF

echo "[3/5] 第一遍完整性检查通过：200/200，verdict=green"
echo "[4/5] 第二遍：specialist，seed_k=12/24"
PASS2_BEFORE="$(mktemp)"
find "$PRED_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$PASS2_BEFORE"
JITTOR_COMPILER_THREADS="$JITTOR_COMPILER_THREADS" "$PYTHON_BIN" \
    "$JITTOR_TASK_RUNNER" --task b_final/_runtime_reproduce_pass2
PASS2_RUN="$(comm -13 "$PASS2_BEFORE" \
    <(find "$PRED_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
    | grep 'b_final_pass2_specialist_cascade_official_b' | tail -1 || true)"
rm -f "$PASS2_BEFORE"
if [ -z "$PASS2_RUN" ]; then
    echo "ERROR: 未找到第二遍新生成的预测目录"
    exit 1
fi

PASS2_DIR="$PRED_ROOT/$PASS2_RUN/pred"
PASS2_MANIFEST="$PRED_ROOT/$PASS2_RUN/manifest.json"
N_PRED2="$(find "$PASS2_DIR/shapenet/00000000" -name denoised.npy 2>/dev/null | wc -l)"
if [ "$N_PRED2" -ne 200 ] || [ ! -f "$PASS2_MANIFEST" ]; then
    echo "ERROR: 第二遍输出不完整：predictions=$N_PRED2 manifest=$PASS2_MANIFEST"
    exit 1
fi
VERDICT2="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["verdict"])' "$PASS2_MANIFEST")"
if [ "$VERDICT2" != "green" ]; then
    echo "ERROR: 第二遍 manifest verdict=$VERDICT2"
    exit 1
fi

echo "[5/5] 第二遍检查通过，生成提交结果包"
cd "$SCRIPT_DIR"
"$PYTHON_BIN" scripts/shared/pack_submission.py \
    --predict-run "outputs/predictions/b_final/$PASS2_RUN" \
    --dataset-test dataset/test_noisy_b \
    --submit-root outputs/submissions

SUBMISSION="$(ls -1t outputs/submissions/b_final/*/result.zip 2>/dev/null | head -1)"
if [ -z "$SUBMISSION" ]; then
    echo "ERROR: 未生成 result.zip"
    exit 1
fi

echo "[PASS] B 榜两遍级联复现完成"
echo "  first pass:  $PASS1_RUN"
echo "  second pass: $PASS2_RUN"
echo "  result:      $SCRIPT_DIR/$SUBMISSION"
echo "  submitted checkpoint score: 81.05 (CD 70.01, P2S 92.10)"
