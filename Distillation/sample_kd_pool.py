#!/usr/bin/env python3
"""
Sample a fixed subset from the decontaminated KD pool.
Samples per source (Big-Math, MATH, NuminaMath) and shuffles the combined pool.
"""

import json
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any


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


def main():
    parser = argparse.ArgumentParser(
        description="Sample fixed KD pool subset from decontaminated data"
    )
    parser.add_argument(
        "--input_path",
        default="data/kd_pool_clean.jsonl",
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output_path",
        default="out/kd_pool_sampled.jsonl",
        help="Path to output sampled JSONL file"
    )
    parser.add_argument(
        "--per_source",
        type=int,
        default=10_000,
        help="Max examples to sample per source"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    print(f"[INFO] Loading data from {args.input_path}")
    data = load_jsonl(args.input_path)
    print(f"[INFO] Loaded {len(data)} total examples")
    
    # Group by source
    by_source = defaultdict(list)
    for ex in data:
        src = ex.get("source", "UNKNOWN")
        by_source[src].append(ex)
    
    # Sample per source
    sampled = []
    for src, pool in sorted(by_source.items()):
        k = min(args.per_source, len(pool))
        chosen = random.sample(pool, k)
        print(f"[INFO] Sampling {k} from source={src} (pool={len(pool)})")
        sampled.extend(chosen)
    
    # Shuffle combined pool
    random.shuffle(sampled)
    
    # Save sampled pool
    save_jsonl(args.output_path, sampled)
    print(f"[INFO] Saved {len(sampled)} sampled examples to {args.output_path}")


if __name__ == "__main__":
    main()

