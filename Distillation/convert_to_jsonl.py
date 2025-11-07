#!/usr/bin/env python3
"""
Convert JSON array datasets to JSONL format for KD pipeline.
Maps fields: {prompt → question, keep source, generate id if missing}
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def load_json(path: str) -> List[Dict[str, Any]]:
    """Load JSON array from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    """Save list of dicts as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_training_data(input_path: str, output_path: str) -> None:
    """
    Convert training dataset: prompt → question, keep source, generate id if missing.
    
    Input format: [{"prompt": "...", "answer": "...", "source": "...", "id": "..."}]
    Output format: {"question": "...", "source": "...", "id": "..."}
    """
    print(f"[INFO] Loading training data from {input_path}")
    data = load_json(input_path)
    
    converted = []
    for idx, item in enumerate(data):
        converted_item = {
            "question": item["prompt"],
            "source": item.get("source", "UNKNOWN"),
            "id": item.get("id", f"train_{idx:06d}")
        }
        converted.append(converted_item)
    
    save_jsonl(output_path, converted)
    print(f"[INFO] Converted {len(converted)} examples to {output_path}")


def convert_benchmark_data(input_path: str, output_path: str) -> None:
    """
    Convert benchmark dataset: prompt → question, keep answer & source, generate id if missing.
    
    Input format: [{"prompt": "...", "answer": "...", "source": "..."}]
    Output format: {"question": "...", "answer": "...", "source": "...", "id": "..."}
    """
    print(f"[INFO] Loading benchmark data from {input_path}")
    data = load_json(input_path)
    
    converted = []
    for idx, item in enumerate(data):
        converted_item = {
            "question": item["prompt"],
            "answer": item.get("answer", ""),
            "source": item.get("source", "UNKNOWN"),
            "id": item.get("id", f"benchmark_{idx:06d}")
        }
        converted.append(converted_item)
    
    save_jsonl(output_path, converted)
    print(f"[INFO] Converted {len(converted)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSON array datasets to JSONL format"
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to input JSON file"
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--type",
        choices=["training", "benchmark"],
        default="training",
        help="Type of dataset (affects field mapping)"
    )
    args = parser.parse_args()
    
    if args.type == "training":
        convert_training_data(args.input_path, args.output_path)
    elif args.type == "benchmark":
        convert_benchmark_data(args.input_path, args.output_path)


if __name__ == "__main__":
    main()

