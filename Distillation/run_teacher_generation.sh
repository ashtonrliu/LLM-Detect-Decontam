#!/bin/bash
# Run teacher generation with vLLM in parallel across multiple GPUs
# Based on the pattern from LLM-Math-Evaluation

# Exit on error
set -e

# Parse arguments
MODEL_PATH=""
INPUT_FILE=""
OUTPUT_FILE=""
NUM_GPUS=3
MAX_TOKENS=512
TEMP=0.0

# Simple argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --input_file)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output_file)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --max_tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --temperature)
            TEMP="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MODEL_PATH" || -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
    echo "Usage: $0 --model_path MODEL --input_file INPUT --output_file OUTPUT [--num_gpus N] [--max_tokens N] [--temperature T]"
    echo ""
    echo "Example:"
    echo "  $0 --model_path ../models/Qwen2.5/Qwen2.5-Math-7B-Instruct \\"
    echo "     --input_file out/kd_pool_sampled.jsonl \\"
    echo "     --output_file out/qwen_teacher_kd.jsonl \\"
    echo "     --num_gpus 3"
    exit 1
fi

# Create logs directory
mkdir -p logs

echo "========================================"
echo "Teacher Generation with vLLM"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "GPUs: $NUM_GPUS"
echo "Max tokens: $MAX_TOKENS"
echo "Temperature: $TEMP"
echo "========================================"
echo ""

# Start time
START_TIME=$(date +%s)

# Launch parallel workers (one per GPU)
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    LOG_FILE="logs/teacher_gen_gpu${i}.log"
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting worker $i on GPU $i"
    echo "  Log: $LOG_FILE"
    
    # Run with CUDA_VISIBLE_DEVICES to pin to specific GPU
    CUDA_VISIBLE_DEVICES=$i conda run -n llm-distil-contamination python -u generate_teacher_solutions.py \
        --model_id "$MODEL_PATH" \
        --input_path "$INPUT_FILE" \
        --output_path "$OUTPUT_FILE" \
        --task_id $i \
        --task_count $NUM_GPUS \
        --max_new_tokens $MAX_TOKENS \
        --temperature $TEMP \
        > "$LOG_FILE" 2>&1 &
    
    PIDS+=($!)
done

echo ""
echo "All workers launched. PIDs: ${PIDS[@]}"
echo "Monitor progress with: tail -f logs/teacher_gen_gpu*.log"
echo "Or use: watch -n 1 'tail -n 5 logs/teacher_gen_gpu*.log'"
echo ""

# Wait for all workers to complete
echo "Waiting for all workers to complete..."
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait $PID; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker $i (PID $PID) completed successfully"
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Worker $i (PID $PID) FAILED"
        FAILED=1
    fi
done

# End time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "========================================"
if [[ $FAILED -eq 0 ]]; then
    echo "All workers completed successfully!"
    echo "Time elapsed: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    
    # Merge output files if multiple GPUs were used
    if [[ $NUM_GPUS -gt 1 ]]; then
        echo ""
        echo "Merging output files..."
        for ((i=0; i<NUM_GPUS; i++)); do
            PARTIAL_FILE="${OUTPUT_FILE%.jsonl}_${i}.jsonl"
            if [[ -f "$PARTIAL_FILE" ]]; then
                cat "$PARTIAL_FILE" >> "$OUTPUT_FILE"
                echo "  Merged: $PARTIAL_FILE"
            fi
        done
        
        # Count total lines
        TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
        echo "Total solutions generated: $TOTAL_LINES"
        
        echo ""
        echo "Output saved to: $OUTPUT_FILE"
    fi
else
    echo "Some workers FAILED. Check logs in logs/teacher_gen_gpu*.log"
    exit 1
fi
echo "========================================"

