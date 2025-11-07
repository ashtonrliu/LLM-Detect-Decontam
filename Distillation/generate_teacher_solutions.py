#!/usr/bin/env python3
"""
Generate step-by-step CoT solutions using teacher models with vLLM.
Creates (input_text, target_text) pairs for knowledge distillation.
Uses vLLM for high-throughput batched inference.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from vllm import LLM, SamplingParams


# Fixed prompt template for all teacher models
PROMPT_TEMPLATE = """You are a math problem solving assistant.
Solve the following problem step by step and put the final answer in \\boxed{{}}.

Problem:
{question}

Solution:
"""


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Save list of dicts as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(question: str) -> str:
    """Build full prompt from question using template."""
    return PROMPT_TEMPLATE.format(question=question)


def main():
    parser = argparse.ArgumentParser(
        description="Generate teacher solutions for knowledge distillation using vLLM"
    )
    parser.add_argument(
        "--model_id",
        required=True,
        help="Path or HF ID of teacher model"
    )
    parser.add_argument(
        "--input_path",
        default="out/kd_pool_sampled.jsonl",
        help="Path to input JSONL file with questions"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to output JSONL file with teacher solutions"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="Task ID for data parallelism (0-indexed)"
    )
    parser.add_argument(
        "--task_count",
        type=int,
        default=1,
        help="Total number of parallel tasks"
    )
    args = parser.parse_args()
    
    # Load input data
    print(f"[INFO] Loading questions from {args.input_path}")
    all_data = load_jsonl(args.input_path)
    print(f"[INFO] Loaded {len(all_data)} total examples")
    
    # Split data for this task
    data = [ex for i, ex in enumerate(all_data) if i % args.task_count == args.task_id]
    print(f"[INFO] Task {args.task_id}/{args.task_count} will process {len(data)} examples")
    
    # Build all prompts
    print("[INFO] Building prompts...")
    prompts = [build_prompt(ex["question"]) for ex in data]
    
    # Initialize vLLM model (single GPU per task, no tensor parallelism)
    print(f"[INFO] Loading teacher model with vLLM from {args.model_id}")
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
        max_model_len=2048,
    )
    
    # Configure sampling parameters
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0.0 else 0.0,
        top_p=1.0 if args.temperature == 0.0 else 0.95,
    )
    
    # Generate all solutions with vLLM (batched automatically)
    print(f"[INFO] Generating {len(prompts)} solutions...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"[INFO] Generation complete! Processing {len(outputs)} outputs...")
    
    # Process outputs
    out_rows = []
    for i, (ex, output) in enumerate(zip(data, outputs)):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        
        # Create output record
        out_rows.append({
            "id": ex["id"],
            "source": ex["source"],
            "input_text": prompt,
            "target_text": generated_text.strip(),
        })
        
        # Progress logging
        if (i + 1) % 100 == 0:
            print(f"[INFO] Task {args.task_id}: Processed {i+1}/{len(data)} outputs")
    
    # Save teacher KD data (with task_id suffix if parallel)
    if args.task_count > 1:
        output_path = args.output_path.replace(".jsonl", f"_{args.task_id}.jsonl")
    else:
        output_path = args.output_path
    
    save_jsonl(output_path, out_rows)
    print(f"[INFO] Saved {len(out_rows)} teacher solutions to {output_path}")
    
    # Show a few examples
    if len(out_rows) > 0:
        print("\n[INFO] Sample outputs:")
        sample_indices = [0, len(out_rows)//2, -1] if len(out_rows) > 2 else [0]
        for idx in sample_indices:
            ex = out_rows[idx]
            print(f"\n--- Example {idx} (ID: {ex['id']}) ---")
            print(f"Input (first 100 chars): {ex['input_text'][:100]}...")
            print(f"Target (first 150 chars): {ex['target_text'][:150]}...")


if __name__ == "__main__":
    main()

