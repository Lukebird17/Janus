#!/bin/bash
set -euo pipefail

# ================================================================
# Stage 0 (Janus-Pro-7B): Collect Activation Statistics
#
# 使用已有的校准数据集，通过 Janus 模型收集激活统计
# (Hessian for GPTQ, channel stats for SmoothQuant/AWQ)
# ================================================================

PYTHON="${PYTHON:-/home/honglianglu/data/.conda/envs/janus/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JANUS_QUANT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODEL_PATH="${MODEL_PATH:-/data/user/honglianglu/Bagel/models/Janus-Pro-7B}"

GPU_IDS="${GPU_IDS:-0}"
SEED="${SEED:-42}"
N="${N:-200}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 复用 Bagel 的校准数据集（格式通用: image_path + question）
CALIB_JSON="${CALIB_JSON:-/data/user/honglianglu/Bagel/quantization/quantization_outputs/csc_sweep/N${N}_seed${SEED}/calibration_subset_N${N}.json}"

# 输出目录: 按 N 和 seed 组织
OUTPUT_DIR="${JANUS_QUANT_ROOT}/quantization_outputs/stage0_N${N}_seed${SEED}"

echo "================================================================"
echo " Stage 0 (Janus-Pro-7B): Collect Activation Statistics"
echo "================================================================"
echo "  Model       : ${MODEL_PATH}"
echo "  Calibration : ${CALIB_JSON}"
echo "  Output      : ${OUTPUT_DIR}"
echo "  GPU         : ${GPU_IDS}"
echo "  N           : ${N}"
echo "================================================================"

if [ ! -f "${CALIB_JSON}" ]; then
    echo "ERROR: Calibration dataset not found: ${CALIB_JSON}"
    echo "Please run Bagel's build_calibration_dataset.py first, or specify CALIB_JSON."
    exit 1
fi

# 检查是否已经有输出
if [ -f "${OUTPUT_DIR}/gptq_hessian_index_latest.json" ]; then
    echo ""
    echo "Stage 0 outputs already exist:"
    ls -la "${OUTPUT_DIR}"/gptq_hessian_index_latest.json
    echo "Skipping. Delete the directory to re-run."
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

${PYTHON} \
    "${JANUS_QUANT_ROOT}/stages/stage0_collect_activations.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --calibration_dataset "${CALIB_JSON}" \
    --seed "${SEED}" \
    2>&1 | tee "${OUTPUT_DIR}/stage0_log.txt"

echo ""
echo "Stage 0 complete. Outputs:"
ls -la "${OUTPUT_DIR}"/*latest*
