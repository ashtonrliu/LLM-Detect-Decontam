# LLM-Detect-Decontam

**Cross-Family Knowledge Distillation with Rigorous Dataset Decontamination**

A complete pipeline for training small language models through knowledge distillation while ensuring zero contamination with evaluation benchmarks. Includes GPT-3-style n-gram overlap detection and comprehensive evaluation framework.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This repository implements a research pipeline that:

1. **Detects and removes dataset contamination** using GPT-3-style n-gram overlap analysis
2. **Generates teacher solutions** from large language models (7B-8B parameters)
3. **Trains student models** via knowledge distillation with LoRA
4. **Evaluates cross-family vs same-family distillation** effectiveness

**Key Research Question:** Does knowledge distillation work across different model architectures (e.g., Llama ← Qwen)?

---

## 🎯 Key Results

- ✅ **Cross-family distillation works**: Both cross-family students significantly outperformed base models
- 🏆 **Best performance**: Same-family + specialized teacher (Qwen→Qwen: 43.8% on MATH-500)
- 🔬 **Teacher quality matters more than architecture**: Llama student from Qwen-Math teacher (23.8%) > Llama student from Llama teacher (18.4%)
- 📊 **Zero contamination confirmed**: Rigorous decontamination ensures evaluation integrity

See [CROSS_FAMILY_DISTILLATION_FINDINGS.md](CROSS_FAMILY_DISTILLATION_FINDINGS.md) for detailed results.

---

## 📁 Repository Structure

```
LLM-Detect-Decontam/
├── Decontamination/              # Dataset contamination detection & removal
│   ├── detect_contamination.py
│   ├── decontaminate_dataset.py
│   ├── verify_contamination.py
│   ├── data/
│   │   ├── benchmark/            # Evaluation benchmarks (MATH-500, AIME-2025)
│   │   └── datasets/             # Training datasets (original & decontaminated)
│   └── results/                  # Contamination detection outputs
│
├── Distillation/                 # Knowledge distillation pipeline
│   ├── convert_to_jsonl.py
│   ├── generate_teacher_solutions.py
│   ├── train_student_sft.py
│   ├── merge_lora_adapter.py
│   ├── eval_math500_vllm.py
│   ├── create_results_csv.py
│   ├── create_visualizations.py
│   ├── run_teacher_generation.sh
│   ├── run_batch_eval.sh
│   ├── data/                     # Processed datasets
│   └── out/                      # Teacher outputs, evaluations, visualizations
│
├── models/                       # Downloaded model checkpoints (gitignored)
├── download_models.py            # Utility to download models from HuggingFace
├── requirements.txt
├── README.md                     # This file
├── REPLICATION.md               # Quick start replication guide
├── METHODOLOGY.md               # Detailed methodology
├── PIPELINE_SUMMARY.md          # Complete technical documentation
└── CROSS_FAMILY_DISTILLATION_FINDINGS.md  # Results and analysis
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n llm-distil python=3.10 -y
conda activate llm-distil

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace (required for Llama models)
huggingface-cli login
# Enter your HuggingFace token when prompted
```

**Note:** You need a HuggingFace account and access token for downloading Llama models. Get your token at: https://huggingface.co/settings/tokens

### Download Models

```bash
# Download all required models (~20GB)
python download_models.py

# Or download specific models
python download_models.py Llama-3.2-1B-Instruct Qwen2.5-Math-7B-Instruct

# List available models
python download_models.py --list
```

**Required Models:**
- `Llama-3.1-8B-Instruct` - Teacher model
- `Llama-3.2-1B-Instruct` - Student base model
- `Qwen2.5-Math-7B-Instruct` - Math-specialized teacher
- `Qwen2.5-1.5B-Instruct` - Student base model

### Complete Pipeline

**Pipeline Overview:**
1. **Decontamination** (30 min): Detect and remove contaminated examples
2. **Data Preparation** (5 min): Convert to JSONL format
3. **Teacher Generation** (1-2 hours): Generate solutions using teacher models
4. **Student Training** (2-4 hours per student): Train via LoRA-based distillation
5. **Evaluation** (1-2 hours): Evaluate on MATH-500 and AIME-2025
6. **Results & Visualization** (5 min): Generate CSV summaries and plots

All steps are detailed in the sections below with copy-paste ready commands.

---

## 📊 Models Evaluated

| Category | Model | Parameters | Description |
|----------|-------|------------|-------------|
| **Teachers** | Qwen2.5-Math-7B-Instruct | 7B | Math-specialized teacher |
| | Llama-3.1-8B-Instruct | 8B | General instruction-tuned teacher |
| **Students (Same-Family)** | qwen_from_qwen | 1.5B | Qwen ← Qwen-Math |
| | llama_instruct_from_llama | 1B | Llama ← Llama |
| **Students (Cross-Family)** | llama_from_qwen | 1B | Llama ← Qwen-Math ✨ |
| | qwen_from_llama | 1.5B | Qwen ← Llama ✨ |
| **Baselines** | Qwen2.5-1.5B-Instruct | 1.5B | Pre-trained baseline |
| | Llama-3.2-1B-Instruct | 1B | Pre-trained baseline |

---

## 🧪 Decontamination Pipeline

### Method: GPT-3-Style N-gram Overlap Detection

Uses [Microsoft Overlapy](https://github.com/microsoft/overlapy) for efficient contamination detection.

**Parameters:**
- N-gram range: 8-13 consecutive words
- Threshold: ANY exact match = contaminated
- Algorithm: Aho-Corasick string matching

### Usage

```bash
cd Decontamination

# 1. Detect contamination
python detect_contamination.py \
    data/datasets/distillation_dataset_sampled_10k.json \
    data/benchmark/math-500.json \
    data/benchmark/aime-2025.json

# 2. Remove contaminated examples
python decontaminate_dataset.py \
    data/datasets/distillation_dataset_sampled_10k.json

# 3. Verify zero contamination
python verify_contamination.py \
    data/datasets/distillation_dataset_sampled_10k_decontaminated.json \
    data/benchmark/math-500.json \
    data/benchmark/aime-2025.json
```

**Results:**
- Original dataset: 30,000 examples
- Contaminated: 1,494 examples (5%)
- Clean dataset: 28,506 examples
- Verification: ✅ 0% contamination

---

## 🎓 Knowledge Distillation Pipeline

### Data Preparation

First, convert the decontaminated dataset to JSONL format:

```bash
cd Distillation

# Convert training data to JSONL
python convert_to_jsonl.py \
    --input_path ../Decontamination/data/datasets/distillation_dataset_sampled_10k_decontaminated.json \
    --output_path data/kd_pool_clean.jsonl \
    --type training

# Convert benchmarks to JSONL
python convert_to_jsonl.py \
    --input_path ../Decontamination/data/benchmark/math-500.json \
    --output_path data/math500.jsonl \
    --type benchmark

python convert_to_jsonl.py \
    --input_path ../Decontamination/data/benchmark/aime-2025.json \
    --output_path data/aime2025.jsonl \
    --type benchmark
```

### Teacher Generation

```bash
# Generate solutions from Qwen teacher
bash run_teacher_generation.sh \
    --model_path ../models/Qwen2.5/Qwen2.5-Math-7B-Instruct \
    --input_file data/kd_pool_clean.jsonl \
    --output_file out/qwen_teacher_kd.jsonl \
    --num_gpus 3 \
    --max_tokens 512 \
    --temperature 0.0

# Generate solutions from Llama teacher
bash run_teacher_generation.sh \
    --model_path ../models/Llama/Llama-3.1-8B-Instruct \
    --input_file data/kd_pool_clean.jsonl \
    --output_file out/llama_teacher_kd.jsonl \
    --num_gpus 3 \
    --max_tokens 512 \
    --temperature 0.0
```

**Note:** If you have only 1 GPU, use `--num_gpus 1` (will take ~3 hours per teacher instead of ~1 hour)

### Student Training

```bash
# Train student via LoRA-based SFT
python train_student_sft.py \
    --student_model_id ../models/Qwen2.5/Qwen2.5-1.5B-Instruct \
    --teacher_kd_path out/qwen_teacher_kd.jsonl \
    --output_dir out/students/qwen_from_qwen \
    --batch_size 4 \
    --num_epochs 1 \
    --learning_rate 3e-5 \
    --max_len 1024 \
    --gradient_checkpointing \
    --lora_rank 16 \
    --lora_alpha 32 \
    --trust_remote_code

# Merge LoRA adapter for inference
python merge_lora_adapter.py \
    --base_model ../models/Qwen2.5/Qwen2.5-1.5B-Instruct \
    --lora_checkpoint out/students/qwen_from_qwen/checkpoint-XXXX \
    --output_dir out/students/qwen_from_qwen_merged \
    --trust_remote_code
```

### Evaluation

```bash
# Evaluate single model
python eval_math500_vllm.py \
    --model_id out/students/qwen_from_qwen_merged \
    --data_path data/math500.jsonl \
    --output_dir out \
    --max_new_tokens 2048 \
    --prefix_ratios "1.0,0.8,0.6,0.4" \
    --trust_remote_code

# Batch evaluation (all models on both benchmarks)
bash run_batch_eval.sh
```

This evaluates all 8 models on both MATH-500 and AIME-2025 with 4 prefix ratios each.

### Results Analysis and Visualization

After evaluation completes, generate comprehensive results:

```bash
# Aggregate all evaluation results into CSV
python create_results_csv.py

# Generate visualization plots
python create_visualizations.py
```

**Outputs:**
- `out/complete_evaluation_results.csv` - All results in CSV format
- `out/visualizations/` - 6 comprehensive plots:
  - `plot1_model_comparison.png` - Model performance comparison
  - `plot2_memorization_analysis.png` - Accuracy vs. prefix ratio
  - `plot3_teacher_student_comparison.png` - Knowledge transfer visualization
  - `plot4_answer_format_quality.png` - Answer formatting adherence
  - `plot5_rouge_memorization.png` - ROUGE-L memorization patterns
  - `plot6_dataset_difficulty_heatmap.png` - MATH-500 vs AIME-2025 difficulty

**View Results:**
```bash
# View aggregated CSV results
cat out/complete_evaluation_results.csv

# View specific model results
cat out/eval_math500_qwen_from_qwen_merged_vllm.json | python -m json.tool
```

---

## 💻 Hardware Requirements

**Minimum:**
- GPU: 1× 24GB VRAM (RTX 3090 / A5000 / A6000)
- CPU: 16 cores
- RAM: 64GB
- Storage: 100GB

**Recommended (used in this project):**
- GPU: 3× 24GB VRAM (RTX 3090)
- CPU: 128 cores
- RAM: 128GB
- Storage: 500GB

---

## 📖 Additional Documentation

Additional methodology and results documentation is available locally but not included in the GitHub repository. Clone and check the local files for:

- **REPLICATION.md** - Detailed step-by-step replication guide with troubleshooting
- **METHODOLOGY.md** - Research methodology and experimental design
- **PIPELINE_SUMMARY.md** - Complete technical documentation (1290 lines)
- **CROSS_FAMILY_DISTILLATION_FINDINGS.md** - Experimental results and comprehensive analysis

The README above contains all essential information to run the complete pipeline.

---

## 📚 Citation

### Contamination Detection Method

```bibtex
@inproceedings{brown2020language,
  title={Language models are few-shot learners},
  author={Brown, Tom and Mann, Benjamin and Ryder, Nick and others},
  booktitle={Advances in Neural Information Processing Systems},
  year={2020}
}
```

### Overlapy Tool

Microsoft Overlapy: https://github.com/microsoft/overlapy

---

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

This is a research repository. If you find issues or have suggestions:
1. Open an issue describing the problem/suggestion
2. For significant changes, discuss first in an issue
3. Submit pull requests with clear descriptions

---

## ⚙️ Technical Stack

- **Python**: 3.10.18
- **PyTorch**: 2.5.1
- **Transformers**: Latest (HuggingFace)
- **vLLM**: 0.7.2 (high-throughput inference)
- **PEFT**: Latest (LoRA implementation)
- **DeepSpeed**: 0.16.3
- **Overlapy**: Microsoft contamination detection library

---

## 📧 Contact

For questions about this research or pipeline, please open an issue on GitHub.

**Last Updated:** November 7, 2025
