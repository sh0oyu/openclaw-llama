#!/usr/bin/env python3
import os
from huggingface_hub import snapshot_download
import torch

def setup():
    print("Setting up OpenClaw with Llama 3.2 1B...")
    
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU")
    
    model_id = "unsloth/Llama-3.2-1B-Instruct"
    
    print(f"Downloading {model_id}...")
    print("This takes 5-10 minutes first time...")
    
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir="./models/llama-3.2-1b",
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Model downloaded!")
    except Exception as e:
        print(f"Download error: {e}")
        print("Will retry on first run.")
    
    print("Setup complete! Run: python openclaw.py")

if __name__ == "__main__":
    setup()
