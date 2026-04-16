#!/bin/bash
set -euo pipefail

# ================================================================
# Stage 0 (Janus-Pro-7B): Collect Activation Statistics
#
# 1. 构建 Janus 自己的校准数据集（Flickr8K → 多类型 VQA）
# 2. 子采样 N 条
# 3. 收集激活统计 (Hessian for GPTQ, channel stats for SmoothQuant/AWQ)
# ================================================================

PYTHON="${PYTHON:-/home/honglianglu/data/.conda/envs/janus/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JANUS_QUANT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODEL_PATH="${MODEL_PATH:-/data/user/honglianglu/Bagel/models/Janus-Pro-7B}"

GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-42}"
N="${N:-200}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Flickr8K 数据（与 Bagel 共用源数据，但校准集独立构建）
FLICKR8K_ROOT="${FLICKR8K_ROOT:-/data/user/honglianglu/Bagel/data/flickr8k}"

# Janus 自己的校准数据目录
CALIB_ROOT="${JANUS_QUANT_ROOT}/quantization_outputs/calibration_data"
FULL_CALIB="${CALIB_ROOT}/calibration_dataset_latest.json"

# 子采样输出
OUTPUT_DIR="${JANUS_QUANT_ROOT}/quantization_outputs/stage0_N${N}_seed${SEED}"
SUBSET_JSON="${OUTPUT_DIR}/calibration_subset_N${N}.json"

echo "================================================================"
echo " Stage 0 (Janus-Pro-7B): Collect Activation Statistics"
echo "================================================================"
echo "  Model       : ${MODEL_PATH}"
echo "  Flickr8K    : ${FLICKR8K_ROOT}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  GPU         : ${GPU_IDS}"
echo "  N           : ${N}"
echo "  Seed        : ${SEED}"
echo "================================================================"

# ================================================================
# Step 0a: 构建完整校准数据集（1000 条 Flickr8K VQA）
# ================================================================
if [ -f "${FULL_CALIB}" ]; then
    echo ""
    echo "[Step 0a] Full calibration dataset already exists: ${FULL_CALIB}"
else
    echo ""
    echo "[Step 0a] Building full calibration dataset (1000 samples) ..."
    if [ ! -d "${FLICKR8K_ROOT}" ]; then
        echo "ERROR: Flickr8K not found: ${FLICKR8K_ROOT}"
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="" ${PYTHON} \
        "${JANUS_QUANT_ROOT}/utils/build_calibration_dataset.py" \
        --flickr8k_root "${FLICKR8K_ROOT}" \
        --output_dir "${CALIB_ROOT}" \
        --num_samples 1000 \
        --seed "${SEED}"
    echo "  Done: ${FULL_CALIB}"
fi

# ================================================================
# Step 0b: 子采样 N 条
# ================================================================
mkdir -p "${OUTPUT_DIR}"

if [ -f "${SUBSET_JSON}" ]; then
    echo ""
    echo "[Step 0b] Subset JSON already exists: ${SUBSET_JSON}"
else
    echo ""
    echo "[Step 0b] Subsampling ${N} from full dataset ..."
    CUDA_VISIBLE_DEVICES="" ${PYTHON} \
        "${JANUS_QUANT_ROOT}/scripts/subsample_calibration.py" \
        --input "${FULL_CALIB}" \
        --output "${SUBSET_JSON}" \
        --num_samples "${N}" \
        --seed "${SEED}"
fi

# ================================================================
# Step 0c: 收集激活统计
# ================================================================
if [ -f "${OUTPUT_DIR}/gptq_hessian_index_latest.json" ]; then
    echo ""
    echo "[Step 0c] Stage 0 outputs already exist:"
    ls -la "${OUTPUT_DIR}"/gptq_hessian_index_latest.json
    echo "Skipping. Delete the directory to re-run."
    exit 0
fi

echo ""
echo "[Step 0c] Collecting activation statistics ..."
${PYTHON} \
    "${JANUS_QUANT_ROOT}/stages/stage0_collect_activations.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --calibration_dataset "${SUBSET_JSON}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUTPUT_DIR}/stage0_log.txt"

echo ""
echo "Stage 0 complete. Outputs:"
ls -la "${OUTPUT_DIR}"/*latest*
