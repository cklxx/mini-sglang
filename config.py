"""Global configuration for sglang-mini demo.

This mirrors the global config hooks in full sglang (model choice + device
selection) in a single lightweight file.
"""
from __future__ import annotations

import os
import torch

MODEL_NAME: str = os.environ.get("MODEL_NAME", "gpt2")
MAX_NEW_TOKENS_DEFAULT: int = 64


def get_device() -> str:
    """Select the best available device.

    Preference order: Apple Silicon MPS > CUDA > CPU.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
