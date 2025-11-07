#!/bin/bash
# Comprehensive batch evaluation script for all 8 models on MATH-500
# Fixed ROUGE-L calculation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
DATASET="math500"
DATA_PATH="/home/ailab/dev/LLM-Detect-Decontam/Decontamination/data/benchmarks/MATH-500.jsonl"
MAX_NEW_TOKENS=2048
TEMPERATURE=0.0

# Define all 8 models
declare -A MODELS
MODELS["qwen_teacher"]="/home/ailab/dev/models/Qwen2.5/Qwen2.5-Math-7B-Instruct"
MODELS["llama_teacher"]="/home/ailab/dev/models/Llama/Llama-3.1-8B-Instruct"
MODELS["qwen_base"]="/home/ailab/dev/models/Qwen2.5/Qwen2.5-1.5B-Instruct"
MODELS["qwen_student"]="out/students/qwen_from_qwen_merged"
MODELS["qwen_cross"]="out/students/qwen_from_llama_merged"
MODELS["llama_base"]="/home/ailab/dev/models/Llama/Llama-3.2-1B-Instruct"
MODELS["llama_student"]="out/students/llama_instruct_from_llama_merged"
MODELS["llama_cross"]="out/students/llama_from_qwen_merged"

# GPU allocation (cycle through 3 GPUs)
GPUS=(0 1 2)
GPU_IDX=0

# Log directory
mkdir -p logs

echo "======================================"
echo "FULL MATH-500 EVALUATION - ALL 8 MODELS"
echo "Fixed ROUGE-L calculation"
echo "======================================"
echo ""
echo "Dataset: $DATA_PATH"
echo "Max tokens: $MAX_NEW_TOKENS"
echo "Temperature: $TEMPERATURE"
echo ""
echo "Models to evaluate:"
for key in "${!MODELS[@]}"; do
    echo "  - $key: ${MODELS[$key]}"
done
echo ""
echo "======================================"
echo ""

# Function to wait for GPU availability
wait_for_gpu() {
    local gpu_id=$1
    while true; do
        # Check if GPU has less than 5GB memory used
        mem_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
        if [ "$mem_used" -lt 5000 ]; then
            echo "[INFO] GPU $gpu_id available (${mem_used}MB used)"
            return 0
        fi
        echo "[INFO] GPU $gpu_id busy (${mem_used}MB used). Waiting 30s..."
        sleep 30
    done
}

# Evaluate each model
PIDS=()
MODEL_KEYS=()

for key in qwen_teacher llama_teacher qwen_base qwen_student qwen_cross llama_base llama_student llama_cross; do
    model_path="${MODELS[$key]}"
    
    # Get next GPU (round-robin)
    GPU_ID=${GPUS[$GPU_IDX]}
    GPU_IDX=$(( (GPU_IDX + 1) % ${#GPUS[@]} ))
    
    # Wait for GPU to be available
    wait_for_gpu $GPU_ID
    
    # Extract model name for output files
    model_name=$(basename "$model_path")
    
    log_file="logs/eval_math500_${key}_$(date +%Y%m%d_%H%M%S).log"
    
    echo "======================================"
    echo "Starting: $key"
    echo "Model: $model_path"
    echo "GPU: $GPU_ID"
    echo "Log: $log_file"
    echo "======================================"
    echo ""
    
    # Run evaluation in background
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    PYTHONUNBUFFERED=1 \
    conda run -n llm-distil-contamination python -u eval_math500_vllm.py \
        --model_name_or_path "$model_path" \
        --data_path "$DATA_PATH" \
        --output_dir "out" \
        --dataset_name "$DATASET" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        2>&1 | tee "$log_file" &
    
    pid=$!
    PIDS+=($pid)
    MODEL_KEYS+=($key)
    
    echo "[INFO] Started $key (PID: $pid) on GPU $GPU_ID"
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
    echo "  - ${MODEL_KEYS[$i]}: PID ${PIDS[$i]}"
done
echo ""
echo "Monitoring progress..."
echo ""

# Wait for all jobs to complete
failed=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    key=${MODEL_KEYS[$i]}
    
    echo "[INFO] Waiting for $key (PID: $pid)..."
    
    if wait $pid; then
        echo "✓ $key completed successfully"
    else
        echo "✗ $key FAILED (exit code: $?)"
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

