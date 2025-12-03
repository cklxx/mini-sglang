"""Global configuration for sglang-mini demo.

This mirrors the global config hooks in full sglang (model choice + device
selection) in a single lightweight file.
"""
from __future__ import annotations

import os
import torch

# Default to a modern, small instruct model for better quality than legacy GPT-2.
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B-Instruct")
# Use a longer default generation budget to surface streaming benefits.
MAX_NEW_TOKENS_DEFAULT: int = 512


def get_device() -> str:
    """Select the best available device.

    Preference order: Apple Silicon MPS > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
