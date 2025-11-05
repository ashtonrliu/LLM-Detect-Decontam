#!/usr/bin/env python3
"""
Decontaminate training dataset by removing all contaminated examples.

This script reads the contamination detection results and creates a clean dataset
by removing all training examples that were flagged across all benchmarks.

Usage:
    python decontaminate_dataset.py <training_dataset>
    
Example:
    python decontaminate_dataset.py data/datasets/distillation_dataset_sampled_10k.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter


def load_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSON dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python decontaminate_dataset.py <training_dataset>")
        print("\nExample:")
        print("  python decontaminate_dataset.py data/datasets/distillation_dataset_sampled_10k.json")
        print("\nNote: This requires contamination detection to have been run first.")
        sys.exit(1)
    
    train_path = Path(sys.argv[1])
    results_dir = Path('results') / 'detection'
    summary_file = results_dir / 'contamination_summary.json'
    
    if not summary_file.exists():
        print(f"ERROR: Contamination detection results not found at {summary_file}")
        print("\nRun detection first:")
        print(f"  python detect_contamination.py {train_path} <benchmark1> [benchmark2] ...")
        sys.exit(1)
    
    print("="*70)
    print("DATASET DECONTAMINATION")
    print("="*70)
    
    # Load contamination results
    print(f"\nLoading contamination detection results...")
    with open(summary_file, 'r') as f:
        contamination_summary = json.load(f)
    
    contaminated_indices = set(contamination_summary['contaminated_train_indices'])
    
    print(f"  Benchmarks analyzed: {len(contamination_summary['benchmarks'])}")
    for bench in contamination_summary['benchmarks']:
        print(f"    - {bench['benchmark']}: {bench['contaminated_test']}/{bench['test_size']} contaminated ({bench['contamination_rate']:.1f}%)")
    print(f"  Total contaminated training examples: {len(contaminated_indices)}")
    
    # Load training data
    print(f"\nLoading training dataset: {train_path}")
    train_data = load_dataset(train_path)
    print(f"  Original size: {len(train_data):,} examples")
    
    # Filter out contaminated examples
    print(f"\nRemoving contaminated examples...")
    clean_data = [
        ex for idx, ex in enumerate(train_data)
        if idx not in contaminated_indices
    ]
    
    # Statistics
    original_count = len(train_data)
    clean_count = len(clean_data)
    removed_count = original_count - clean_count
    removed_pct = (removed_count / original_count) * 100
    
    print(f"\n{'='*70}")
    print("DECONTAMINATION SUMMARY")
    print(f"{'='*70}")
    
    print(f"\nDataset:")
    print(f"  Original:       {original_count:>6,} examples")
    print(f"  Decontaminated: {clean_count:>6,} examples ({100-removed_pct:.1f}%)")
    print(f"  Removed:        {removed_count:>6,} examples ({removed_pct:.1f}%)")
    
    # By source - original
    original_sources = Counter([ex['source'] for ex in train_data])
    print(f"\nOriginal by source:")
    for source in sorted(original_sources.keys()):
        count = original_sources[source]
        pct = (count / original_count) * 100
        print(f"  {source:12}: {count:>6,} ({pct:5.1f}%)")
    
    # By source - clean
    clean_sources = Counter([ex['source'] for ex in clean_data])
    print(f"\nDecontaminated by source:")
    for source in sorted(clean_sources.keys()):
        count = clean_sources[source]
        pct = (count / clean_count) * 100
        print(f"  {source:12}: {count:>6,} ({pct:5.1f}%)")
    
    # Removed by source
    print(f"\nRemoved by source:")
    for source in sorted(original_sources.keys()):
        original = original_sources[source]
        clean = clean_sources.get(source, 0)
        removed = original - clean
        removed_pct_source = (removed / original) * 100 if original > 0 else 0
        print(f"  {source:12}: {removed:>6,} ({removed_pct_source:5.1f}% of {source})")
    
    # Save decontaminated dataset
    output_path = train_path.parent / f"{train_path.stem}_decontaminated.json"
    
    print(f"\nSaving decontaminated dataset...")
    with open(output_path, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    
    # Save clean indices for reference
    clean_indices_file = results_dir / 'clean_train_indices.json'
    clean_indices = [idx for idx in range(len(train_data)) if idx not in contaminated_indices]
    with open(clean_indices_file, 'w') as f:
        json.dump(clean_indices, f, indent=2)
    print(f"  Clean indices: {clean_indices_file}")
    
    print(f"\n{'='*70}")
    print("✅ DECONTAMINATION COMPLETE")
    print(f"{'='*70}")
    
    print(f"\nNext step: Verify zero contamination")
    print(f"  python verify_contamination.py {output_path} <benchmark1> [benchmark2] ...")


if __name__ == "__main__":
    main()

