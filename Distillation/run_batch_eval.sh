#!/bin/bash
# Comprehensive batch evaluation script
# All 8 models on MATH-500 only
# With corrected ROUGE-L methodology (Wu et al. 2025)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
DATASET_NAME="math500"
DATA_PATH="data/math500.jsonl"
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

echo "======================================"
echo "BATCH EVALUATION - ALL 8 MODELS"
echo "Dataset: MATH-500 (corrected ROUGE-L)"
echo "======================================"
echo ""

# Define all 8 models with their paths
declare -a MODELS=(
    "/home/ailab/dev/models/Qwen2.5/Qwen2.5-Math-7B-Instruct"
    "/home/ailab/dev/models/Llama/Llama-3.1-8B-Instruct"
    "/home/ailab/dev/models/Qwen2.5/Qwen2.5-1.5B"
    "out/students/qwen_from_qwen_merged"
    "out/students/qwen_from_llama_merged"
    "/home/ailab/dev/models/Llama/Llama-3.2-1B"
    "out/students/llama_instruct_from_llama_merged"
    "out/students/llama_from_qwen_merged"
)

declare -a MODEL_NAMES=(
    "Qwen2.5-Math-7B-Instruct"
    "Llama-3.1-8B-Instruct"
    "Qwen2.5-1.5B"
    "qwen_from_qwen_merged"
    "qwen_from_llama_merged"
    "Llama-3.2-1B"
    "llama_instruct_from_llama_merged"
    "llama_from_qwen_merged"
)

# GPU allocation (cycle through 3 GPUs)
GPUS=(0 1 2)
GPU_IDX=0

# Log directory
mkdir -p logs

# Function to wait for GPU availability
wait_for_gpu() {
    local gpu_id=$1
    while true; do
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        if [ "$mem_used" -lt 5000 ]; then
            echo "[INFO] GPU $gpu_id available (${mem_used}MB used)"
            return 0
        fi
        echo "[INFO] GPU $gpu_id busy (${mem_used}MB used). Waiting 30s..."
        sleep 30
    done
}

# Track running jobs
PIDS=()
GPU_ASSIGNMENTS=()

echo "Starting evaluations..."
echo ""

# Launch all 8 models
for i in "${!MODELS[@]}"; do
    model_path="${MODELS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    
    # Get next GPU (round-robin)
    GPU_ID=${GPUS[$GPU_IDX]}
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))
    
    # Wait for GPU to be available
    wait_for_gpu $GPU_ID
    
    log_file="logs/eval_${DATASET_NAME}_${model_name}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "======================================"
    echo "Starting: ${model_name}"
    echo "Model: ${model_path}"
    echo "GPU: ${GPU_ID}"
    echo "Log: ${log_file}"
    echo "======================================"
    echo ""
    
    # Run evaluation in background
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    PYTHONUNBUFFERED=1 \
    conda run -n llm-distil-contamination python -u eval_math500_vllm.py \
        --model_id "$model_path" \
        --data_path "$DATA_PATH" \
        --output_dir "out" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        2>&1 | tee "$log_file" &
    
    pid=$!
    PIDS+=($pid)
    GPU_ASSIGNMENTS+=("${model_name}:GPU${GPU_ID}")
    
    echo "[INFO] Started ${model_name} (PID: $pid) on GPU $GPU_ID"
    echo ""
    
    # Small delay to stagger starts
    sleep 10
done

echo "======================================"
echo "All evaluation jobs started!"
echo "======================================"
echo ""
echo "Running jobs:"
for i in "${!PIDS[@]}"; do
    echo "  - ${GPU_ASSIGNMENTS[$i]}: PID ${PIDS[$i]}"
done
echo ""
echo "Monitoring progress..."
echo ""

# Wait for all jobs to complete
failed=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    assignment=${GPU_ASSIGNMENTS[$i]}
    
    echo "[INFO] Waiting for ${assignment} (PID: $pid)..."
    
    if wait $pid; then
        echo "✓ ${assignment} completed successfully"
    else
        exit_code=$?
        echo "✗ ${assignment} FAILED (exit code: $exit_code)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "======================================"
echo "EVALUATION COMPLETE"
echo "======================================"
echo "Total models: ${#PIDS[@]}"
echo "Failed: $failed"
echo ""

if [ $failed -eq 0 ]; then
    echo "✓ All evaluations completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: python create_results_csv.py"
    echo "  2. Run: python create_visualizations.py"
else
    echo "⚠ Some evaluations failed. Check logs in logs/ directory."
    exit 1
fi
