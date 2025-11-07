#!/usr/bin/env python3
"""
Train student models using teacher-generated solutions via LoRA-based SFT.
Implements sequence-level knowledge distillation (plain cross-entropy on teacher outputs).

Usage example:

python train_student_sft.py \
  --student_model_id models/Qwen2.5-1.5B-Instruct \
  --teacher_kd_path out/qwen_teacher_kd.jsonl \
  --output_dir out/students/qwen_from_qwen \
  --batch_size 4 \
  --num_epochs 2
"""

import argparse
import json
from pathlib import Path
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# ---------------
# Tokenization
# ---------------

def make_tokenize_fn(tokenizer, max_len: int = 1024):
    """
    Create tokenization function that concatenates input_text + target_text.
    Applies loss over the full sequence.
    """
    def tokenize_fn(batch):
        input_texts: List[str] = batch["input_text"]
        target_texts: List[str] = batch["target_text"]

        # Concatenate input and target for each example
        texts = [
            inp + tgt
            for inp, tgt in zip(input_texts, target_texts)
        ]

        tok = tokenizer(
            texts,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )

        # Labels = input_ids for standard causal LM
        # (Trainer will handle shifting internally)
        tok["labels"] = tok["input_ids"].copy()
        return tok

    return tokenize_fn


# ---------------
# LoRA target modules
# ---------------

def get_default_lora_target_modules(model) -> list:
    """
    Choose sensible LoRA target modules based on model type.
    Works for Llama 3.x and Qwen2.5-style models.
    """
    model_type = getattr(model.config, "model_type", "")
    model_type = (model_type or "").lower()

    # LLaMA-style & Qwen2.5 both expose q_proj/k_proj/v_proj/o_proj etc.
    # This set is known to work well in many examples.
    target = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # If you ever need to special-case later, you can branch here,
    # e.g. if "qwen" in model_type: use a slightly different list.

    return target


# ---------------
# Main
# ---------------

def main():
    parser = argparse.ArgumentParser(
        description="Train student model with LoRA-based SFT on teacher solutions"
    )
    parser.add_argument(
        "--student_model_id",
        required=True,
        help="Path or HF ID of student model (e.g., models/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--teacher_kd_path",
        required=True,
        help="Path to teacher KD JSONL file (e.g., out/qwen_teacher_kd.jsonl)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save trained model (e.g., out/students/qwen_from_qwen)",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="LoRA dropout",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing for memory savings",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading model/tokenizer (needed for some Qwen variants)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # -----------------
    # 1) Load KD dataset
    # -----------------
    print(f"[INFO] Loading teacher KD dataset from {args.teacher_kd_path}")
    ds = load_dataset("json", data_files=args.teacher_kd_path)["train"]
    print(f"[INFO] Loaded {len(ds)} training examples")

    # -----------------
    # 2) Tokenizer
    # -----------------
    print(f"[INFO] Loading tokenizer from {args.student_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_id,
        trust_remote_code=args.trust_remote_code,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("[INFO] pad_token was None; set pad_token to eos_token")

    # Tokenize dataset
    print("[INFO] Tokenizing dataset...")
    tokenize_fn = make_tokenize_fn(tokenizer, max_len=args.max_len)
    tokenized_ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds.column_names,
        desc="Tokenizing",
    )

    # -----------------
    # 3) Base student model
    # -----------------
    print(f"[INFO] Loading student model from {args.student_model_id}")
    
    # Check if we're in distributed mode (torchrun)
    # If so, don't use device_map (DDP handles device placement)
    import os
    is_distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    
    if is_distributed:
        print(f"[INFO] Distributed training detected (WORLD_SIZE={os.environ.get('WORLD_SIZE')})")
        print("[INFO] Loading model without device_map (DDP will handle placement)")
        model = AutoModelForCausalLM.from_pretrained(
            args.student_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        print("[INFO] Single-GPU training, using device_map='auto'")
        model = AutoModelForCausalLM.from_pretrained(
            args.student_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=args.trust_remote_code,
        )

    if args.gradient_checkpointing:
        print("[INFO] Enabling gradient checkpointing")
        model.gradient_checkpointing_enable()

    # -----------------
    # 4) LoRA config
    # -----------------
    target_modules = get_default_lora_target_modules(model)
    print(f"[INFO] LoRA target_modules: {target_modules}")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    print(f"[INFO] Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------
    # 5) Training setup
    # -----------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",              # no wandb/tensorboard by default
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,   # we already pruned columns in map()
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds,
    )

    # -----------------
    # 6) Train
    # -----------------
    print("[INFO] Starting training...")
    trainer.train()

    # -----------------
    # 7) Save
    # -----------------
    print(f"[INFO] Saving model and tokenizer to {output_dir}")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print("[INFO] Training complete!")


if __name__ == "__main__":
    main()
