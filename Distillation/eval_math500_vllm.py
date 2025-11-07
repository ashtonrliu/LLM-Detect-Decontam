#!/usr/bin/env python3
"""
Fast evaluation using vLLM for batched inference.

Evaluate a model on MATH-500 with full + partial prompts using vLLM for speed.

Example (base model):
  python eval_math500_vllm.py \
    --model_id ../models/Qwen2.5/Qwen2.5-1.5B \
    --data_path data/math500.jsonl \
    --output_dir out \
    --trust_remote_code

Example (merged PEFT model):
  python eval_math500_vllm.py \
    --model_id out/students/qwen_from_qwen_merged \
    --data_path data/math500.jsonl \
    --output_dir out \
    --trust_remote_code

Note: For PEFT/LoRA models, merge them first using the --merge_peft flag in training,
      or use the merge_lora_adapter.py script.
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any

from vllm import LLM, SamplingParams
from datasets import load_dataset
from rouge_score import rouge_scorer


PROMPT_TEMPLATE = """You are a math problem solving assistant.
Solve the following problem step by step and put the final answer in \\boxed{{}}.

Problem:
{question}

Solution:
"""


def build_prompt(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def truncate_question(q: str, ratio: float) -> str:
    """
    Truncate question by character ratio (as in partial-prompt setup).
    """
    if ratio >= 1.0:
        return q
    n = max(1, int(len(q) * ratio))
    return q[:n]


def extract_boxed_answer(text: str) -> str:
    """
    Extract content from the *last* \boxed{...} in text.
    Handles nested braces correctly by counting brace depth.
    If none found, returns empty string.
    """
    # Find all occurrences of \boxed{
    pattern = r"\\boxed\{"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return ""
    
    # Start from the last \boxed{
    start_pos = matches[-1].end()
    
    # Count braces to find the matching closing brace
    brace_count = 1
    i = start_pos
    while i < len(text) and brace_count > 0:
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
        i += 1
    
    if brace_count == 0:
        # Found matching brace
        return text[start_pos:i-1].strip()
    
    # No matching brace found
    return ""


def normalize_answer(ans: str) -> str:
    """
    Normalize math answers for comparison:
    - strip whitespace
    - remove surrounding $ ... $
    - remove surrounding \boxed{...}
    - strip trailing period
    """
    a = ans.strip()
    # remove $ signs
    a = a.replace("$", "").strip()

    # remove \boxed{...} wrapper if still present
    boxed_pat = r"^\\boxed\{(.*)\}$"
    m = re.match(boxed_pat, a)
    if m:
        a = m.group(1).strip()

    # strip trailing period
    a = a.rstrip(".")

    # collapse spaces
    a = " ".join(a.split())
    return a


def safe_model_name(model_id: str) -> str:
    """
    Turn a model_id or local path into a filesystem-safe short name.
    """
    name = model_id
    # strip trailing slash
    if name.endswith("/"):
        name = name[:-1]
    # keep last path component
    name = name.split("/")[-1]
    # replace weird chars
    name = name.replace(":", "_")
    return name


def main():
    parser = argparse.ArgumentParser("Fast vLLM-based MATH-500 evaluation")
    parser.add_argument(
        "--model_id",
        required=True,
        help="HF ID or local path to model",
    )
    parser.add_argument(
        "--data_path",
        default="data/math500.jsonl",
        help="Path to MATH-500 JSONL file",
    )
    parser.add_argument(
        "--output_dir",
        default="out",
        help="Directory to save metrics JSON",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--prefix_ratios",
        type=str,
        default="1.0,0.8,0.6,0.4",
        help="Comma-separated list of prefix ratios, e.g. '1.0,0.8,0.6,0.4'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, evaluate only the first N examples (for quick tests)",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.0-1.0)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)",
    )
    args = parser.parse_args()

    # Parse prefix ratios
    prefix_ratios = [float(x) for x in args.prefix_ratios.split(",")]
    prefix_ratios = sorted(prefix_ratios, reverse=True)
    print(f"[INFO] Prefix ratios: {prefix_ratios}")

    # Load dataset
    print(f"[INFO] Loading MATH-500 from {args.data_path}")
    ds = load_dataset("json", data_files=args.data_path)["train"]
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))
        print(f"[INFO] Limiting evaluation to first {len(ds)} examples")
    
    total = len(ds)
    print(f"[INFO] Total examples to evaluate: {total}")

    # Load vLLM model
    print(f"[INFO] Loading model with vLLM: {args.model_id}")
    print("[INFO] This may take 1-2 minutes...")
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=2048,
        max_num_seqs=256,  # Increase parallel batch size for better GPU utilization
        trust_remote_code=args.trust_remote_code,
    )
    print("[INFO] Model loaded successfully!")

    # Configure sampling
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.temperature > 0.0 else 0.0,
        top_p=1.0 if args.temperature == 0.0 else 0.95,
    )
    print(f"[INFO] Sampling params: max_tokens={args.max_new_tokens}, temperature={args.temperature}")

    # Metrics containers
    correct_counts: Dict[float, int] = {r: 0 for r in prefix_ratios}
    no_boxed_counts: Dict[float, int] = {r: 0 for r in prefix_ratios}
    rougeL_sums: Dict[float, float] = {r: 0.0 for r in prefix_ratios}
    all_predictions: List[Dict[str, Any]] = []
    
    # ROUGE scorer for text-level memorization detection
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # OPTIMIZATION: Build ALL prompts for ALL ratios at once (single batch call)
    print(f"\n[INFO] Building prompts for all {len(prefix_ratios)} prefix ratios...")
    all_prompts = []
    prompt_mapping = []  # Track which (example_idx, ratio) each prompt corresponds to
    
    for ratio in prefix_ratios:
        for idx, ex in enumerate(ds):
            truncated_q = truncate_question(ex["question"], ratio)
            all_prompts.append(build_prompt(truncated_q))
            prompt_mapping.append((idx, ratio))
    
    total_prompts = len(all_prompts)
    print(f"[INFO] Built {total_prompts} prompts ({total} examples × {len(prefix_ratios)} ratios)")
    
    # Generate ALL at once with vLLM (MAXIMUM SPEED!)
    print(f"[INFO] Generating {total_prompts} solutions with vLLM (single batched call)...")
    all_outputs = llm.generate(all_prompts, sampling_params)
    print(f"[INFO] Generation complete!")
    
    # Initialize prediction storage
    for idx, ex in enumerate(ds):
        all_predictions.append({
            "id": ex.get("id", f"ex_{idx}"),
            "gold_answer_raw": ex["answer"],
            "gold_answer_norm": normalize_answer(ex["answer"]),
            "per_prefix": {}
        })
    
    # Process results
    print(f"[INFO] Processing {total_prompts} results...")
    for output_idx, output in enumerate(all_outputs):
        ex_idx, ratio = prompt_mapping[output_idx]
        ex = ds[ex_idx]
        
        full_q = ex["question"]
        gold = ex["answer"]
        norm_gold = normalize_answer(gold)
        truncated_q = truncate_question(full_q, ratio)
        
        # Extract generated text
        generated_text = output.outputs[0].text
        
        # Extract answer
        pred_raw = extract_boxed_answer(generated_text)
        pred_norm = normalize_answer(pred_raw) if pred_raw else ""
        is_correct = (pred_norm != "" and pred_norm == norm_gold)
        
        if is_correct:
            correct_counts[ratio] += 1
        
        # Track answers without boxed format
        if pred_raw == "":
            no_boxed_counts[ratio] += 1
        
        # Compute ROUGE-L for memorization detection (Wu et al. 2025 methodology)
        # Gold tail = the part of the question that was NOT shown to the model
        # CRITICAL: Truncate generated text to match gold_tail length
        # This checks if the FIRST N words of generation match the hidden question
        gold_tail = full_q[len(truncated_q):]
        
        # For ratio 1.0, gold_tail is empty (no hidden text to memorize)
        if len(gold_tail) > 0:
            # Truncate generated text to same word count as gold_tail
            target_words = gold_tail.split()
            generated_words = generated_text.split()
            truncated_generation = " ".join(generated_words[:len(target_words)])
            
            rouge_scores = scorer.score(
                target=gold_tail,
                prediction=truncated_generation,  # Truncated to match length!
            )
            rougeL_f = rouge_scores["rougeL"].fmeasure
        else:
            rougeL_f = 0.0  # No tail to compare for 100% prefix
        
        rougeL_sums[ratio] += rougeL_f
        
        # Store result for this ratio
        all_predictions[ex_idx]["per_prefix"][ratio] = {
            "truncated_question": truncated_q,
            "pred_answer_raw": pred_raw,
            "pred_answer_norm": pred_norm,
            "correct": bool(is_correct),
            "generated_length": len(generated_text),
            "rougeL_f": rougeL_f,
            "gold_tail_length": len(gold_tail),
        }
        
        # Progress reporting (every 500 prompts processed)
        if (output_idx + 1) % 500 == 0:
            progress_pct = (output_idx + 1) / total_prompts * 100
            print(f"  Progress: {output_idx+1}/{total_prompts} ({progress_pct:.1f}%)")
    
    # Report results for each ratio
    print(f"\n[INFO] Results by prefix ratio:")
    for ratio in prefix_ratios:
        acc = correct_counts[ratio] / total * 100 if total > 0 else 0.0
        no_boxed_pct = no_boxed_counts[ratio] / total * 100 if total > 0 else 0.0
        avg_rouge = rougeL_sums[ratio] / total if total > 0 else 0.0
        print(f"  {ratio:.1f}: Acc={correct_counts[ratio]}/{total} ({acc:.2f}%), ROUGE-L={avg_rouge:.4f}, No \\boxed{{}}={no_boxed_counts[ratio]} ({no_boxed_pct:.1f}%)")

    # Aggregate metrics
    accuracies = {
        str(ratio): (correct_counts[ratio] / total if total > 0 else 0.0)
        for ratio in prefix_ratios
    }
    
    avg_rougeL = {
        str(ratio): (rougeL_sums[ratio] / total if total > 0 else 0.0)
        for ratio in prefix_ratios
    }
    
    no_boxed_ratios = {
        str(ratio): (no_boxed_counts[ratio] / total if total > 0 else 0.0)
        for ratio in prefix_ratios
    }

    model_name_safe = safe_model_name(args.model_id)
    
    # Extract dataset name from data_path (e.g., "math500" or "aime2025")
    dataset_name = Path(args.data_path).stem  # Gets filename without extension
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"eval_{dataset_name}_{model_name_safe}_vllm.json"
    preds_path = output_dir / f"eval_{dataset_name}_{model_name_safe}_vllm_preds.jsonl"

    metrics = {
        "model_id": args.model_id,
        "data_path": args.data_path,
        "num_examples": total,
        "prefix_ratios": prefix_ratios,
        "correct_counts": {str(k): int(v) for k, v in correct_counts.items()},
        "accuracies": accuracies,
        "avg_rougeL_f": avg_rougeL,
        "no_boxed_counts": {str(k): int(v) for k, v in no_boxed_counts.items()},
        "no_boxed_ratios": no_boxed_ratios,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "inference_engine": "vllm",
    }

    # Save metrics
    print(f"\n[INFO] Saving results...")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Save per-example predictions
    with open(preds_path, "w", encoding="utf-8") as f:
        for rec in all_predictions:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Model: {args.model_id}")
    print(f"Total examples: {total}")
    print("\nResults by prefix ratio:")
    for ratio in prefix_ratios:
        acc = accuracies[str(ratio)]
        count = correct_counts[ratio]
        rouge = avg_rougeL[str(ratio)]
        no_boxed = no_boxed_counts[ratio]
        no_boxed_pct = no_boxed_ratios[str(ratio)]
        print(f"  {ratio:.1f}: Accuracy={count}/{total} ({acc*100:.2f}%) | ROUGE-L={rouge:.4f} | No \\boxed{{}}={no_boxed} ({no_boxed_pct*100:.1f}%)")
    print("="*80)

    print(f"\n[INFO] Saved metrics to {metrics_path}")
    print(f"[INFO] Saved predictions to {preds_path}")
    print("[INFO] Done.")


if __name__ == "__main__":
    main()

