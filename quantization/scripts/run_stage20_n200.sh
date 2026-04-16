#!/bin/bash
set -euo pipefail

# ================================================================
# Stage 20 (Janus-Pro-7B): Per-Layer CKA Search + Evaluation
#
# 使用 N=200 校准数据，先运行 stage0 收集激活（如果没有），
# 然后运行 stage20 per-layer CKA 搜索 + stage3 评测。
# ================================================================

PYTHON="${PYTHON:-/home/honglianglu/data/.conda/envs/janus/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JANUS_QUANT_ROOT="$(dirname "${SCRIPT_DIR}")"
MODEL_PATH="${MODEL_PATH:-/data/user/honglianglu/Bagel/models/Janus-Pro-7B}"

GPU_IDS="${GPU_IDS:-7}"
MAX_MEM="${MAX_MEM:-40GiB}"
SEED=42
N=200
CKA_NUM_SAMPLES=200
RUN_DATE=$(date +%Y%m%d)

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 校准数据（复用 Bagel 的）
CALIB_JSON="/data/user/honglianglu/Bagel/quantization/quantization_outputs/csc_sweep/N${N}_seed${SEED}/calibration_subset_N${N}.json"

# Stage 0 输出
STATS_DIR="${JANUS_QUANT_ROOT}/quantization_outputs/stage0_N${N}_seed${SEED}"
HESSIAN_IDX="${STATS_DIR}/gptq_hessian_index_latest.json"
SMOOTH_STATS="${STATS_DIR}/smoothquant_stats_latest.pt"
AWQ_STATS="${STATS_DIR}/awq_stats_latest.pt"

# Stage 20 输出
OUTPUT_ROOT="${JANUS_QUANT_ROOT}/quantization_outputs/stage20_n${N}"
SEARCH_DIR="${OUTPUT_ROOT}/search"

BENCHMARKS="${BENCHMARKS:-mme mmvp mmmu geneval}"

echo "================================================================"
echo " Stage 20 (Janus-Pro-7B): Per-Layer CKA Search"
echo "================================================================"
echo "  Model       : ${MODEL_PATH}"
echo "  Calibration : ${CALIB_JSON}"
echo "  GPU         : ${GPU_IDS}"
echo "  CKA samples : ${CKA_NUM_SAMPLES}"
echo "  Benchmarks  : ${BENCHMARKS}"
echo "  Output      : ${OUTPUT_ROOT}"
echo "================================================================"

# ================================================================
# Step 0: Ensure activation stats exist (run stage0 if needed)
# ================================================================
if [ ! -f "${HESSIAN_IDX}" ]; then
    echo ""
    echo "[Step 0] Activation stats not found. Running stage0..."
    bash "${JANUS_QUANT_ROOT}/scripts/run_stage0.sh"
fi

for f in "${CALIB_JSON}" "${HESSIAN_IDX}" "${SMOOTH_STATS}" "${AWQ_STATS}"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: Required file not found: $f"
        exit 1
    fi
done

mkdir -p "${SEARCH_DIR}"

# ================================================================
# Step 1: Stage 20 Search
# ================================================================
if ls "${SEARCH_DIR}"/stage20_search_results_*.json 1>/dev/null 2>&1; then
    echo ""
    echo "[Step 1] Stage 20 search results already exist:"
    ls -la "${SEARCH_DIR}"/stage20_search_results_*.json
else
    echo ""
    echo "[Step 1] Running Stage 20 per-layer CKA search ..."
    ${PYTHON} \
        "${JANUS_QUANT_ROOT}/stages/stage20_largecalib_search.py" \
        --model_path "${MODEL_PATH}" \
        --output_dir "${SEARCH_DIR}" \
        --calibration_dataset "${CALIB_JSON}" \
        --gptq_hessian_index "${HESSIAN_IDX}" \
        --smoothquant_stats "${SMOOTH_STATS}" \
        --awq_stats "${AWQ_STATS}" \
        --gpu_ids "${GPU_IDS}" \
        --max_mem_per_gpu "${MAX_MEM}" \
        --cka_num_samples "${CKA_NUM_SAMPLES}" \
        --seed "${SEED}" \
        --run_date "${RUN_DATE}_N${N}" \
        2>&1 | tee "${OUTPUT_ROOT}/search_log_${RUN_DATE}.txt"
fi

# ================================================================
# Step 2: Find exported config
# ================================================================
QUANT_CONFIG=$(ls -t "${SEARCH_DIR}"/../configs/stage20_largecalib_w4a4_*.json 2>/dev/null | head -1 || true)
if [ -z "${QUANT_CONFIG}" ]; then
    echo "ERROR: Cannot find quantization config."
    exit 1
fi
echo ""
echo "[Step 2] Config: ${QUANT_CONFIG}"

# ================================================================
# Step 3: Evaluation
# ================================================================
CONFIG_NAME="stage20_n${N}_seed${SEED}_${RUN_DATE}"
EVAL_DIR="${OUTPUT_ROOT}/eval/${CONFIG_NAME}"

echo ""
echo "[Step 3] Running evaluation: ${BENCHMARKS} ..."
${PYTHON} \
    "${JANUS_QUANT_ROOT}/stages/stage3_test.py" \
    --model_path "${MODEL_PATH}" \
    --stage2_config "${QUANT_CONFIG}" \
    --gptq_hessian_index "${HESSIAN_IDX}" \
    --smoothquant_stats "${SMOOTH_STATS}" \
    --awq_stats "${AWQ_STATS}" \
    --output_dir "${EVAL_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --benchmarks ${BENCHMARKS} \
    2>&1 | tee "${OUTPUT_ROOT}/eval_log_${RUN_DATE}.txt"

echo ""
echo "================================================================"
echo " Stage 20 Complete (N=${N})"
echo "================================================================"
for f in "${EVAL_DIR}"/*/results.txt; do
    if [ -f "$f" ]; then
        echo "--- $(dirname "$f" | xargs basename) ---"
        cat "$f"
        echo ""
    fi
done
if [ -f "${EVAL_DIR}/geneval_results/summary.txt" ]; then
    echo "--- geneval_results ---"
    cat "${EVAL_DIR}/geneval_results/summary.txt"
    echo ""
fi
