#!/usr/bin/env python3
"""
Detect contamination between training dataset and benchmark(s).

Usage:
    python detect_contamination.py <training_dataset> <benchmark1> [benchmark2] [...]
    
Example:
    python detect_contamination.py \\
        data/datasets/distillation_dataset_sampled_10k.json \\
        data/benchmark/math-500.json \\
        data/benchmark/aime-2025.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
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


def detect_benchmark_contamination(
    benchmark_name: str,
    benchmark_texts: List[str],
    train_texts: List[str],
    train_data: List[Dict[str, Any]],
    n_workers: int
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Run contamination detection for a single benchmark.
    
    Returns:
        (summary_dict, match_details_list)
    """
    print(f"\n{'='*70}")
    print(f"Detecting contamination: {benchmark_name}")
    print(f"{'='*70}")
    
    # Tokenize
    print(f"Tokenizing {len(benchmark_texts)} test examples...")
    testset_tokens = [simple_tokenizer(t) for t in benchmark_texts]
    
    print(f"Tokenizing {len(train_texts):,} training examples...")
    dataset_tokens = [simple_tokenizer(t) for t in train_texts]
    
    # Create test set
    testset = OverlapyTestSet(
        name=benchmark_name.lower().replace('-', '_').replace('.', '_'),
        examples=testset_tokens,
    )
    
    print(f"Running Overlapy (n-gram range: {testset.min_n}-{testset.max_n})...")
    
    # Run detection
    over = Overlapy(testsets=[testset], dataset=dataset_tokens, n_workers=n_workers)
    matches = over.run()
    
    # Process matches
    print("Processing matches...")
    matches_for_test = list(testset.get_matches(matches))
    
    test_to_train_matches = {}
    for test_idx, ngram, position_in_test in matches_for_test:
        if test_idx not in test_to_train_matches:
            test_to_train_matches[test_idx] = set()
        if ngram in matches:
            for train_match in matches[ngram]:
                train_idx = train_match[0] if isinstance(train_match, tuple) else train_match
                test_to_train_matches[test_idx].add(train_idx)
    
    # Build results
    contaminated_test = set(test_to_train_matches.keys())
    contaminated_train = set()
    match_details = []
    
    for test_idx, train_indices in test_to_train_matches.items():
        for train_idx in train_indices:
            contaminated_train.add(train_idx)
            match_details.append({
                'benchmark': benchmark_name,
                'test_idx': test_idx,
                'test_prompt': benchmark_texts[test_idx],
                'train_idx': train_idx,
                'train_id': train_data[train_idx]['id'],
                'train_source': train_data[train_idx]['source'],
                'train_prompt': train_texts[train_idx],
            })
    
    # Summary
    contamination_rate = len(contaminated_test) / len(benchmark_texts) * 100
    
    summary = {
        'benchmark': benchmark_name,
        'test_size': len(benchmark_texts),
        'contaminated_test': len(contaminated_test),
        'contamination_rate': contamination_rate,
        'contaminated_train_indices': sorted(list(contaminated_train)),
        'num_contaminated_train': len(contaminated_train),
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Test examples: {len(benchmark_texts)}")
    print(f"  Contaminated: {len(contaminated_test)} ({contamination_rate:.1f}%)")
    print(f"  Training examples affected: {len(contaminated_train)}")
    
    if match_details:
        sources = Counter([m['train_source'] for m in match_details])
        print(f"  By source: {dict(sources)}")
    
    return summary, match_details


def main():
    if len(sys.argv) < 3:
        print("Usage: python detect_contamination.py <training_dataset> <benchmark1> [benchmark2] ...")
        print("\nExample:")
        print("  python detect_contamination.py \\")
        print("      data/datasets/distillation_dataset_sampled_10k.json \\")
        print("      data/benchmark/math-500.json \\")
        print("      data/benchmark/aime-2025.json")
        sys.exit(1)
    
    train_path = Path(sys.argv[1])
    benchmark_paths = [Path(p) for p in sys.argv[2:]]
    
    print("="*70)
    print("CONTAMINATION DETECTION")
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
    
    # Detect contamination for each benchmark
    n_workers = max(1, cpu_count() - 1)
    all_summaries = []
    all_matches = []
    all_contaminated_train = set()
    
    for benchmark_path in benchmark_paths:
        benchmark_data = load_dataset(benchmark_path)
        benchmark_texts = extract_prompts(benchmark_data)
        benchmark_name = benchmark_path.stem
        
        summary, matches = detect_benchmark_contamination(
            benchmark_name,
            benchmark_texts,
            train_texts,
            train_data,
            n_workers
        )
        
        all_summaries.append(summary)
        all_matches.extend(matches)
        all_contaminated_train.update(summary['contaminated_train_indices'])
    
    # Save results
    results_dir = Path('results') / 'detection'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Combined summary
    combined_summary = {
        'training_dataset': str(train_path),
        'training_size': len(train_texts),
        'benchmarks': all_summaries,
        'total_contaminated_train': len(all_contaminated_train),
        'contaminated_train_indices': sorted(list(all_contaminated_train)),
    }
    
    summary_file = results_dir / 'contamination_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(combined_summary, f, indent=2)
    
    matches_file = results_dir / 'contamination_matches.jsonl'
    with open(matches_file, 'w') as f:
        for match in all_matches:
            f.write(json.dumps(match) + '\n')
    
    # Final report
    print(f"\n{'='*70}")
    print("DETECTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {results_dir}")
    print(f"  - {summary_file.name}")
    print(f"  - {matches_file.name}")
    
    print(f"\nSummary:")
    for summary in all_summaries:
        print(f"  {summary['benchmark']:15} {summary['contaminated_test']:3}/{summary['test_size']:3} contaminated ({summary['contamination_rate']:.1f}%)")
    print(f"\nTotal unique contaminated training examples: {len(all_contaminated_train)}")
    print(f"Percentage of training data: {len(all_contaminated_train)/len(train_texts)*100:.2f}%")
    
    print(f"\nNext step: Run decontamination")
    print(f"  python decontaminate_dataset.py {train_path}")


if __name__ == "__main__":
    main()

