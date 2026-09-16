#!/usr/bin/env bash
# 全流程复现脚本：从原始数据到提交用 result.zip。
#
# 用法（在 a_board/ 目录下执行）：
#   bash run_all.sh full      从零训练两个阶段，拟合第三阶段系数，再推理打包（约 48 小时）
#   bash run_all.sh infer     跳过训练，直接用 checkpoints/ 下的权重推理打包（约 2.2 小时）
#
# 前置条件：
#   1. 已按 requirements.txt 或 environment.yaml 装好依赖；
#   2. dataset/ 已放在 A 榜快照目录下，结构见 README.md；
#   3. 有一块 24GB 显存的 GPU（4090 实测可用）。
#
# 每一步的产物落在快照目录的 outputs/ 下，目录名固定，可从失败步骤重跑。

set -euo pipefail

MODE="${1:-}"
if [[ "$MODE" != "full" && "$MODE" != "infer" ]]; then
    echo "用法: bash run_all.sh {full|infer}" >&2
    exit 2
fi

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CODE_ROOT"

PY="${PYTHON:-python}"
RUNS="outputs/runs/a_final"
PREDS="outputs/predictions/a_final"

log() { echo -e "\n========== $* ==========\n"; }

# 数据集必须先就位，否则后面每一步都会以难懂的方式失败
REQUIRED_DATA=(dataset/test_noisy)
if [[ "$MODE" == "full" ]]; then
    REQUIRED_DATA+=(dataset/train)
fi
for d in "${REQUIRED_DATA[@]}"; do
    if [[ ! -d "$d" ]]; then
        echo "[FAIL] 缺少 $d ，请先按 README.md 放置数据集" >&2
        exit 1
    fi
done

# 第三阶段方向场的系数。full 模式用自己重训权重上拟合出的结果；infer 模式用
# 07_apply_direction_controller.py 里的冻结值，即最终提交所用的系数。
CONTROLLER_MODEL_ARG=()

if [[ "$MODE" == "full" ]]; then
    log "步骤 1/7  生成第二阶段用的固定 noisy/clean 点云对（CPU，约 24 分钟）"
    $PY scripts/pipeline/01_prepare_train_pairs.py \
        --source-datalist starter_code/datalist/train_full15k.txt \
        --datalist-out starter_code/datalist/train_full15k_generated.txt \
        --train-root dataset/train \
        --out-root dataset/a_final_train_full15k \
        --samples 15732 --sample-policy sequential --seed 2026 \
        --allow-mock-overlap --log-every 500

    log "步骤 2/7  第一阶段：基础去噪网络从零训练 100 epoch（约 19 小时）"
    (cd starter_code && $PY run.py --task configs/task/repro/01_train_base.yaml --seed 123)

    log "步骤 3/7  第一阶段推理：官方测试集 200 样本（约 1.1 小时）"
    (cd starter_code && $PY run.py --task configs/task/repro/02_predict_base_official.yaml)

    log "步骤 4/7  第一阶段推理：2000 个训练样本，作为第二阶段的训练输入（约 8.2 小时）"
    $PY scripts/pipeline/03_generate_firstpass_cache.py \
        --datalist starter_code/datalist/specialist_train2000.txt \
        --base-ckpt "../$RUNS/repro_base_train/checkpoints/pdlts_light_99.pkl" \
        --merged-run-id "repro_firstpass_cache_merged" \
        --chunk-size 150

    CACHE_DIR="$PREDS/repro_firstpass_cache_merged/pred"
    if [[ ! -d "$CACHE_DIR" ]]; then
        echo "[FAIL] 找不到合并后的第一阶段输出目录: $CACHE_DIR" >&2
        exit 1
    fi
    echo "第二阶段训练输入: $CACHE_DIR"

    log "步骤 5/7  预计算评分归一化权重（CPU，约 20 分钟）"
    $PY scripts/pipeline/04_precompute_score_weights.py \
        --datalist starter_code/datalist/specialist_train2000.txt \
        --train-root dataset/a_final_train_full15k \
        --run-id repro_score_weights

    log "步骤 6/7  第二阶段训练：主训练 50ep + 全量微调 12ep + 评分归一化微调 12ep（约 17 小时）"
    $PY scripts/pipeline/05_train_specialist.py --stage main \
        --train-datalist starter_code/datalist/specialist_train1600.txt \
        --firstpass-cache "$CACHE_DIR" \
        --load-ckpt "$RUNS/repro_base_train/checkpoints/pdlts_light_99.pkl" \
        --run-id repro_specialist_step1_main

    $PY scripts/pipeline/05_train_specialist.py --stage finetune \
        --train-datalist starter_code/datalist/specialist_train2000.txt \
        --firstpass-cache "$CACHE_DIR" \
        --load-ckpt "$RUNS/repro_specialist_step1_main/specialist_step1_main.pkl" \
        --run-id repro_specialist_step2_finetune

    $PY scripts/pipeline/06_train_specialist_scorenorm.py \
        --train-datalist starter_code/datalist/specialist_train2000.txt \
        --firstpass-cache "$CACHE_DIR" \
        --load-ckpt "$RUNS/repro_specialist_step2_finetune/specialist_step2_finetune.pkl" \
        --cd-noisy-sidecar "outputs/diagnostics/a_final/repro_score_checkpoints/cd_noisy_sidecar.json" \
        --run-id repro_specialist_step3_scorenorm

    log "步骤 7/7  拟合第三阶段方向场系数（第二阶段推理 52 形状 + CPU 拟合，约 20 分钟）"
    $PY scripts/pipeline/07a_fit_direction_controller.py \
        --firstpass-cache "$CACHE_DIR" \
        --stage2-ckpt "../$RUNS/repro_specialist_step3_scorenorm/specialist_step3_scorenorm.pkl" \
        --out "$RUNS/repro_controller_fit"

    # 从零重训的上游输出与最终提交所用不同，方向场系数要用刚拟合出来的这一组，
    # 而不是 07_apply_direction_controller.py 里那组冻结值。
    CONTROLLER_MODEL_ARG=(--model-json "$RUNS/repro_controller_fit/controller_model.json")
else
    log "infer 模式：用 checkpoints/ 下的现成权重，跳过训练"
    mkdir -p "$RUNS/repro_base_train/checkpoints" "$RUNS/repro_specialist_step3_scorenorm"
    cp checkpoints/stage1_base_ep99.pkl "$RUNS/repro_base_train/checkpoints/pdlts_light_99.pkl"
    cp checkpoints/stage2_specialist_final.pkl "$RUNS/repro_specialist_step3_scorenorm/specialist_step3_scorenorm.pkl"

    log "步骤 1/2  第一阶段推理：官方测试集 200 样本（约 1.1 小时）"
    (cd starter_code && $PY run.py --task configs/task/repro/02_predict_base_official.yaml)
fi

log "第二阶段推理：官方测试集 200 样本，输入为第一阶段输出（约 1.1 小时）"
(cd starter_code && $PY run.py --task configs/task/repro/06_predict_specialist_official.yaml)

log "第三阶段：几何方向场后处理（CPU，约 1 分钟）"
$PY scripts/pipeline/07_apply_direction_controller.py \
    --source "$PREDS/repro_specialist_official/pred" \
    --out "$PREDS/repro_final_official/pred" \
    ${CONTROLLER_MODEL_ARG[@]+"${CONTROLLER_MODEL_ARG[@]}"}

log "打包提交结果"
$PY scripts/shared/pack_submission.py \
    --predict-run "$PREDS/repro_final_official" \
    --submit-id "repro_a_final_submit"

echo
echo "完成。提交文件："
echo "  outputs/submissions/a_final/repro_a_final_submit/result.zip"
echo
echo "同目录下还有 check_shapes.txt（逐样本形状检查）与 zip_list.txt（zip 内部路径清单）。"
