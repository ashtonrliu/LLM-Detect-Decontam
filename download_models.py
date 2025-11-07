#!/usr/bin/env python3
"""
Download required models for LLM-Detect-Decontam.
This script downloads LLaMA and Qwen models from HuggingFace.

Usage:
    python download_models.py                           # Download all models
    python download_models.py Llama-3.2-1B              # Download specific model
    python download_models.py Llama-3.2-1B Qwen2.5-1.5B # Download multiple models
    python download_models.py --list                    # List available models
"""

import sys
from huggingface_hub import snapshot_download
from pathlib import Path

# Base directory for models
MODELS_DIR = Path(__file__).parent / "models"

# Model configurations: maps model name to (repo_id, family, local_name)
AVAILABLE_MODELS = {
    "Llama-3.1-8B-Instruct": ("meta-llama/Llama-3.1-8B-Instruct", "Llama", "Llama-3.1-8B-Instruct"),
    "Llama-3.2-1B": ("meta-llama/Llama-3.2-1B", "Llama", "Llama-3.2-1B"),
    "Llama-3.2-1B-Instruct": ("meta-llama/Llama-3.2-1B-Instruct", "Llama", "Llama-3.2-1B-Instruct"),
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B", "Qwen2.5", "Qwen2.5-1.5B"),
    "Qwen2.5-Math-7B-Instruct": ("Qwen/Qwen2.5-Math-7B-Instruct", "Qwen2.5", "Qwen2.5-Math-7B-Instruct"),
}


def download_model(repo_id: str, local_dir: Path):
    """Download a model from HuggingFace."""
    print(f"\n{'='*80}")
    print(f"Downloading: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"{'='*80}\n")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"✓ Successfully downloaded {repo_id}")
        return True
    except Exception as e:
        print(f"✗ Error downloading {repo_id}: {e}")
        return False


def list_models():
    """List all available models."""
    print("\nAvailable models:")
    print("=" * 80)
    for name, (repo_id, family, local_name) in AVAILABLE_MODELS.items():
        local_dir = MODELS_DIR / family / local_name
        status = "✓ Downloaded" if local_dir.exists() and any(local_dir.iterdir()) else "⊙ Not downloaded"
        print(f"  {name:<30} {status}")
        print(f"    Repository: {repo_id}")
        print(f"    Location:   models/{family}/{local_name}")
        print()
    print("=" * 80)


def main():
    """Main function to download models based on command line arguments."""
    # Handle --list flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--list", "-l", "list"]:
        list_models()
        return
    
    # Handle --help flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--help", "-h", "help"]:
        print(__doc__)
        list_models()
        return
    
    # Determine which models to download
    if len(sys.argv) > 1:
        # Specific models requested
        requested_models = sys.argv[1:]
        models_to_download = []
        
        for model_name in requested_models:
            if model_name in AVAILABLE_MODELS:
                repo_id, family, local_name = AVAILABLE_MODELS[model_name]
                models_to_download.append((model_name, repo_id, family, local_name))
            else:
                print(f"✗ Unknown model: {model_name}")
                print(f"\nAvailable models:")
                for name in AVAILABLE_MODELS.keys():
                    print(f"  - {name}")
                print(f"\nUse 'python download_models.py --list' for more details")
                return
        
        print(f"Downloading {len(models_to_download)} specific model(s)...")
    else:
        # Default: download all models
        models_to_download = [
            (name, *details) for name, details in AVAILABLE_MODELS.items()
        ]
        print(f"Downloading all {len(models_to_download)} models...")
    
    print(f"Models will be saved to: {MODELS_DIR}\n")
    
    downloaded = 0
    skipped = 0
    failed = []
    
    for model_name, repo_id, family, local_name in models_to_download:
        family_dir = MODELS_DIR / family
        family_dir.mkdir(parents=True, exist_ok=True)
        local_dir = family_dir / local_name
        
        # Check if already exists
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"⊙ Skipping {model_name} (already exists at {local_dir})")
            skipped += 1
            continue
        
        if download_model(repo_id, local_dir):
            downloaded += 1
        else:
            failed.append(repo_id)
    
    # Summary
    print(f"\n{'='*80}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*80}")
    print(f"Total models requested: {len(models_to_download)}")
    print(f"Successfully downloaded: {downloaded}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFailed models:")
        for model in failed:
            print(f"  - {model}")
    else:
        print("\n✓ All requested models processed successfully!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

