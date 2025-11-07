#!/usr/bin/env python3
"""
Aggregate all evaluation results into a single CSV file.
"""

import json
import csv
from pathlib import Path

def main():
    output_dir = Path("out")
    
    # Find all evaluation JSON files
    eval_files = sorted(output_dir.glob("eval_*_vllm.json"))
    
    if not eval_files:
        print("[ERROR] No evaluation files found!")
        return
    
    print(f"[INFO] Found {len(eval_files)} evaluation files")
    
    # Collect all results
    results = []
    
    for eval_file in eval_files:
        print(f"[INFO] Processing {eval_file.name}")
        
        with open(eval_file, 'r') as f:
            data = json.load(f)
        
        # Parse model and dataset from filename
        # Format: eval_{dataset}_{model}_vllm.json
        parts = eval_file.stem.replace("eval_", "").replace("_vllm", "").split("_")
        
        # First part is dataset
        dataset = parts[0]
        # Rest is model name
        model = "_".join(parts[1:])
        
        # Get total examples
        total = data["num_examples"]
        
        # Extract metrics for each prefix ratio
        for ratio_str in data["prefix_ratios"]:
            ratio_key = str(float(ratio_str))
            row = {
                "model": model,
                "dataset": dataset,
                "prefix_ratio": ratio_key,
                "accuracy": data["accuracies"][ratio_key],
                "correct": data["correct_counts"][ratio_key],
                "total": total,
                "avg_rougeL_f": data["avg_rougeL_f"][ratio_key],
                "no_boxed_count": data["no_boxed_counts"][ratio_key],
                "no_boxed_ratio": data["no_boxed_ratios"][ratio_key],
            }
            results.append(row)
    
    # Sort results
    results.sort(key=lambda x: (x["model"], x["dataset"], float(x["prefix_ratio"])))
    
    # Write to CSV
    csv_path = output_dir / "complete_evaluation_results.csv"
    fieldnames = [
        "model", "dataset", "prefix_ratio", "accuracy", "correct", "total",
        "avg_rougeL_f", "no_boxed_count", "no_boxed_ratio"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n[INFO] ✓ CSV saved to {csv_path}")
    print(f"[INFO] Total rows: {len(results)}")
    
    # Print summary table
    print("\n" + "="*100)
    print("SUMMARY: Models × Datasets (Accuracy at prefix_ratio=1.0)")
    print("="*100)
    
    # Get unique models and datasets
    models = sorted(set(r["model"] for r in results))
    datasets = sorted(set(r["dataset"] for r in results))
    
    # Create summary table for prefix_ratio=1.0
    print(f"{'Model':<50} | {'Dataset':<15} | {'Accuracy':<10} | {'Correct/Total':<15}")
    print("-" * 100)
    
    for model in models:
        for dataset in datasets:
            # Find the result for this model-dataset combination at prefix_ratio=1.0
            match = [r for r in results if r["model"] == model and r["dataset"] == dataset and r["prefix_ratio"] == "1.0"]
            if match:
                r = match[0]
                acc_pct = r["accuracy"] * 100
                print(f"{model:<50} | {dataset:<15} | {acc_pct:>9.2f}% | {r['correct']:>3}/{r['total']:<3}")
            else:
                print(f"{model:<50} | {dataset:<15} | {'MISSING':<10} | {'':>15}")
    
    print("="*100)
    
    # Validate we have all expected combinations
    expected_count = len(models) * len(datasets) * 4  # 4 prefix ratios
    if len(results) == expected_count:
        print(f"\n[INFO] ✓ All {expected_count} expected results present!")
    else:
        print(f"\n[WARNING] Expected {expected_count} results, but found {len(results)}")
        print(f"[WARNING] Models: {len(models)}, Datasets: {len(datasets)}, Prefix ratios: 4")
    
    print(f"\n[INFO] Complete results saved to: {csv_path}")

if __name__ == "__main__":
    main()

