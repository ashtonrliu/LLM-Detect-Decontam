# LLM-Detect-Decontam

Dataset contamination detection and removal pipeline for Large Language Models using GPT-3-style n-gram overlap analysis.

## Repository Structure

```
LLM-Detect-Decontam/
├── Decontamination/          # Contamination detection & removal pipeline
│   ├── detect_contamination.py
│   ├── decontaminate_dataset.py
│   ├── verify_contamination.py
│   ├── data/
│   │   ├── benchmark/        # Evaluation benchmarks
│   │   └── datasets/         # Training datasets
│   └── results/              # Detection results
└── Distillation/             # (Future: distillation pipeline)
```

---

## Decontamination Pipeline

### Prerequisites

```bash
conda create -n llm-detect-decontam python=3.10 -y
conda activate llm-detect-decontam
pip install overlapy
```

### 3-Step Pipeline

#### Step 1: Detect Contamination

Identify which training examples overlap with benchmark(s).

```bash
cd Decontamination
python detect_contamination.py <training_dataset> <benchmark1> [benchmark2] ...
```

**Example:**
```bash
python detect_contamination.py \
    data/datasets/distillation_dataset_sampled_10k.json \
    data/benchmark/math-500.json \
    data/benchmark/aime-2025.json
```

**Output:** `results/detection/contamination_summary.json`

#### Step 2: Remove Contamination

Remove all contaminated examples.

```bash
python decontaminate_dataset.py <training_dataset>
```

**Example:**
```bash
python decontaminate_dataset.py data/datasets/distillation_dataset_sampled_10k.json
```

**Output:** `data/datasets/<name>_decontaminated.json`

#### Step 3: Verify Zero Contamination

Confirm the cleaned dataset has no contamination.

```bash
python verify_contamination.py <decontaminated_dataset> <benchmark1> [benchmark2] ...
```

**Example:**
```bash
python verify_contamination.py \
    data/datasets/distillation_dataset_sampled_10k_decontaminated.json \
    data/benchmark/math-500.json \
    data/benchmark/aime-2025.json
```

**Output:** `results/verification/verification_summary.json`

---

## Quick Reference

### Complete Pipeline Example

```bash
# Activate environment
conda activate llm-detect-decontam
cd Decontamination

# 1. Detect
python detect_contamination.py \
    data/datasets/my_training_data.json \
    data/benchmark/my_benchmark.json

# 2. Decontaminate
python decontaminate_dataset.py data/datasets/my_training_data.json

# 3. Verify
python verify_contamination.py \
    data/datasets/my_training_data_decontaminated.json \
    data/benchmark/my_benchmark.json
```

### Data Format

All JSON files must have this structure:

```json
[
  {
    "prompt": "Problem text...",
    "answer": "Solution...",
    "source": "dataset_name",
    "id": "unique_id"
  }
]
```

---

## Method

Uses **GPT-3-style n-gram overlap detection**:
- Tokenization: Simple whitespace splitting
- N-gram size: 8-13 consecutive words
- Threshold: ANY exact match = contaminated
- Algorithm: Aho-Corasick (efficient string matching)

Based on: Brown et al., "Language Models are Few-Shot Learners" (2020), Appendix C

---

## Current Datasets

### Benchmarks
- `math-500.json` - 500 MATH problems
- `aime-2025.json` - 30 AIME 2025 problems

### Training Datasets
- `distillation_dataset_sampled_10k.json` - 30k problems (original)
- `distillation_dataset_decontaminated.json` - 28,507 problems (MATH-500 clean)
- `distillation_dataset_fully_decontaminated.json` - 28,506 problems (both benchmarks clean)

---

## Results

### MATH-500
- Original: 82% contaminated (410/500)
- After decontamination: 0% contaminated ✅

### AIME-2025
- Original: 20% contaminated (6/30)
- After decontamination: 0% contaminated ✅

### Training Data
- Removed: 1,494 examples (5%)
- Retained: 28,506 examples (95%)
- All sources preserved (MATH, Big-Math, NuminaMath)

---

## Citation

If using this pipeline in research:

**Method:** Brown et al., "Language Models are Few-Shot Learners" (2020), Appendix C  
**Tool:** Overlapy - https://github.com/microsoft/overlapy

---

## License

MIT License - See LICENSE file for details.

