#!/usr/bin/env python3
"""
Verify that a dataset has zero contamination with given benchmarks.

This re-runs contamination detection on a cleaned dataset to confirm
that all contamination has been successfully removed.

Usage:
    python verify_contamination.py <training_dataset> <benchmark1> [benchmark2] ...
    
Example:
    python verify_contamination.py \\
        data/datasets/distillation_dataset_sampled_10k_decontaminated.json \\
        data/benchmark/math-500.json \\
        data/benchmark/aime-2025.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter
from multiprocessing import cpu_count

try:
    from overlapy import OverlapyTestSet, Overlapy
except ImportError:
    print("ERROR: Overlapy not installed. Run: pip install overlapy")
    sys.exit(1)


def load_dataset(filepath: Path) -> List[Dict[str, Any]]:
    """Load a JSON dataset."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_prompts(dataset: List[Dict[str, Any]]) -> List[str]:
    """Extract prompt field from dataset entries."""
    return [item['prompt'] for item in dataset]


def simple_tokenizer(text: str) -> List[str]:
    """Simple tokenizer for Overlapy."""
    return text.lower().split()


def verify_benchmark(
    benchmark_name: str,
    benchmark_texts: List[str],
    train_texts: List[str],
    n_workers: int
) -> tuple:
    """
    Verify zero contamination for a benchmark.
    
    Returns:
        (num_contaminated_test, num_contaminated_train)
    """
    print(f"\n  Verifying: {benchmark_name}")
    
    # Tokenize
    testset_tokens = [simple_tokenizer(t) for t in benchmark_texts]
    dataset_tokens = [simple_tokenizer(t) for t in train_texts]
    
    # Create test set
    testset = OverlapyTestSet(
        name=benchmark_name.lower().replace('-', '_').replace('.', '_'),
        examples=testset_tokens,
    )
    
    # Run detection
    over = Overlapy(testsets=[testset], dataset=dataset_tokens, n_workers=n_workers)
    matches = over.run()
    
    # Process matches
    matches_for_test = list(testset.get_matches(matches))
    
    test_to_train_matches = {}
    for test_idx, ngram, position_in_test in matches_for_test:
        if test_idx not in test_to_train_matches:
            test_to_train_matches[test_idx] = set()
        if ngram in matches:
            for train_match in matches[ngram]:
                train_idx = train_match[0] if isinstance(train_match, tuple) else train_match
                test_to_train_matches[test_idx].add(train_idx)
    
    contaminated_test = len(test_to_train_matches)
    contaminated_train = len(set().union(*test_to_train_matches.values())) if test_to_train_matches else 0
    
    return contaminated_test, contaminated_train


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_contamination.py <training_dataset> <benchmark1> [benchmark2] ...")
        print("\nExample:")
        print("  python verify_contamination.py \\")
        print("      data/datasets/distillation_dataset_sampled_10k_decontaminated.json \\")
        print("      data/benchmark/math-500.json \\")
        print("      data/benchmark/aime-2025.json")
        sys.exit(1)
    
    train_path = Path(sys.argv[1])
    benchmark_paths = [Path(p) for p in sys.argv[2:]]
    
    print("="*70)
    print("CONTAMINATION VERIFICATION")
    print("="*70)
    print(f"\nTraining dataset: {train_path}")
    print(f"Benchmarks: {len(benchmark_paths)}")
    for bp in benchmark_paths:
        print(f"  - {bp}")
    
    # Load training data
    print(f"\nLoading training dataset...")
    train_data = load_dataset(train_path)
    train_texts = extract_prompts(train_data)
    print(f"  Loaded: {len(train_texts):,} examples")
    
    source_dist = Counter([item['source'] for item in train_data])
    print(f"  Sources: {dict(source_dist)}")
    
    # Verify each benchmark
    print(f"\nRunning verification (this will take a few minutes)...")
    n_workers = max(1, cpu_count() - 1)
    
    all_results = []
    total_contaminated_test = 0
    total_contaminated_train = 0
    
    for benchmark_path in benchmark_paths:
        benchmark_data = load_dataset(benchmark_path)
        benchmark_texts = extract_prompts(benchmark_data)
        benchmark_name = benchmark_path.stem
        
        contam_test, contam_train = verify_benchmark(
            benchmark_name,
            benchmark_texts,
            train_texts,
            n_workers
        )
        
        all_results.append({
            'benchmark': benchmark_name,
            'test_size': len(benchmark_texts),
            'contaminated_test': contam_test,
            'contaminated_train': contam_train,
        })
        
        total_contaminated_test += contam_test
        total_contaminated_train = max(total_contaminated_train, contam_train)
    
    # Final report
    print(f"\n{'='*70}")
    print("VERIFICATION RESULTS")
    print(f"{'='*70}")
    
    print(f"\nDataset: {train_path.name}")
    print(f"Size: {len(train_texts):,} examples")
    
    print(f"\nPer-Benchmark Results:")
    all_clean = True
    for result in all_results:
        contam_rate = (result['contaminated_test'] / result['test_size']) * 100
        status = "✅" if result['contaminated_test'] == 0 else "❌"
        print(f"  {status} {result['benchmark']:15} {result['contaminated_test']:3}/{result['test_size']:3} contaminated ({contam_rate:.1f}%)")
        if result['contaminated_test'] > 0:
            all_clean = False
    
    print(f"\nOverall:")
    print(f"  Total test examples contaminated: {total_contaminated_test}")
    print(f"  Total train examples contaminated: {total_contaminated_train}")
    
    # Save verification results
    results_dir = Path('results') / 'verification'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    verification_summary = {
        'training_dataset': str(train_path),
        'training_size': len(train_texts),
        'benchmarks': all_results,
        'all_clean': all_clean,
    }
    
    summary_file = results_dir / 'verification_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(verification_summary, f, indent=2)
    
    print(f"\nResults saved to: {summary_file}")
    
    print(f"\n{'='*70}")
    if all_clean:
        print("✅ VERIFICATION PASSED")
        print(f"{'='*70}")
        print("\nDataset is fully decontaminated!")
        print("All benchmarks show 0% contamination.")
        print("\nReady for training and evaluation!")
    else:
        print("❌ VERIFICATION FAILED")
        print(f"{'='*70}")
        print("\nContamination still detected.")
        print("\nPossible causes:")
        print("  1. Decontamination was not run")
        print("  2. Different benchmarks than detection")
        print("  3. Dataset was modified after decontamination")
        print("\nRecommendation: Re-run decontamination pipeline")


if __name__ == "__main__":
    main()

