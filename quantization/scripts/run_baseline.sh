#!/bin/bash
set -euo pipefail

# ================================================================
# Janus-Pro-7B BF16 Baseline Evaluation
#
# 运行原始未量化模型的 MME / MMVP / MMMU / GenEval 基线评测。
# 使用一个空的量化配置 (empty JSON) 来跳过量化步骤。
# ================================================================

PYTHON="${PYTHON:-/home/honglianglu/data/.conda/envs/janus/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JANUS_QUANT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODEL_PATH="${MODEL_PATH:-/data/user/honglianglu/Bagel/models/Janus-Pro-7B}"

GPU_IDS="${GPU_IDS:-1,2,3}"
BENCHMARKS="${BENCHMARKS:-mme mmvp mmmu geneval}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

OUTPUT_DIR="${JANUS_QUANT_ROOT}/quantization_outputs/baseline_bf16"

echo "================================================================"
echo " Janus-Pro-7B BF16 Baseline Evaluation"
echo "================================================================"
echo "  Model      : ${MODEL_PATH}"
echo "  GPU        : ${GPU_IDS}"
echo "  Benchmarks : ${BENCHMARKS}"
echo "  Output     : ${OUTPUT_DIR}"
echo "================================================================"

mkdir -p "${OUTPUT_DIR}"

# 创建空的量化配置（绕过量化步骤）
EMPTY_CONFIG="${OUTPUT_DIR}/empty_config.json"
echo '{}' > "${EMPTY_CONFIG}"

${PYTHON} \
    "${JANUS_QUANT_ROOT}/stages/stage3_test.py" \
    --model_path "${MODEL_PATH}" \
    --stage2_config "${EMPTY_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --benchmarks ${BENCHMARKS} \
    2>&1 | tee "${OUTPUT_DIR}/baseline_log.txt"

echo ""
echo "================================================================"
echo " Baseline Complete"
echo "================================================================"
for f in "${OUTPUT_DIR}"/*/results.txt; do
    if [ -f "$f" ]; then
        echo "--- $(dirname "$f" | xargs basename) ---"
        cat "$f"
        echo ""
    fi
done
if [ -f "${OUTPUT_DIR}/geneval_results/summary.txt" ]; then
    echo "--- geneval_results ---"
    cat "${OUTPUT_DIR}/geneval_results/summary.txt"
    echo ""
fi
