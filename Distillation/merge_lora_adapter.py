#!/usr/bin/env python3
"""
Merge LoRA adapter weights into base model for vLLM evaluation.

Usage:
  python merge_lora_adapter.py \
    --base_model ../models/Qwen2.5/Qwen2.5-1.5B \
    --lora_checkpoint out/students/qwen_from_qwen/checkpoint-28506 \
    --output_dir out/students/qwen_from_qwen_merged \
    --trust_remote_code
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM


def main():
    parser = argparse.ArgumentParser("Merge LoRA adapter into base model")
    parser.add_argument(
        "--base_model",
        required=True,
        help="Path to base model (for reference, will be loaded from PEFT config)",
    )
    parser.add_argument(
        "--lora_checkpoint",
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save merged model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading",
    )
    args = parser.parse_args()

    print("="*80)
    print("LoRA Adapter Merger")
    print("="*80)

    # Load PEFT model (which includes base + adapter)
    print(f"\n[INFO] Loading PEFT model from {args.lora_checkpoint}...")
    print("[INFO] This may take a minute...")
    peft_model = AutoPeftModelForCausalLM.from_pretrained(
        args.lora_checkpoint,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
    )
    print("[INFO] PEFT model loaded successfully!")

    # Merge adapter into base weights
    print("[INFO] Merging LoRA adapter into base model weights...")
    merged_model = peft_model.merge_and_unload()
    print("[INFO] Merge complete!")

    # Save merged model
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Saving merged model to {output_path}...")
    merged_model.save_pretrained(str(output_path))
    print("[INFO] Model saved!")

    # Also save tokenizer (from base model)
    print(f"[INFO] Saving tokenizer from base model...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer.save_pretrained(str(output_path))
    print("[INFO] Tokenizer saved!")

    print("\n" + "="*80)
    print("MERGE COMPLETE")
    print("="*80)
    print(f"Merged model saved to: {output_path}")
    print(f"You can now use this with vLLM for fast evaluation:")
    print(f"  python eval_math500_vllm.py --model_id {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()

