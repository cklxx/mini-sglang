"""Lightweight runtime helpers used across backends and engine."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any, Tuple

import torch

logger = logging.getLogger(__name__)


def configure_torch(device: str) -> None:
    """Set a few global torch knobs for inference."""
    precision = os.getenv("MATMUL_PRECISION")
    if precision:
        torch.set_float32_matmul_precision(precision)
    elif device in {"cuda", "mps"}:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if os.getenv("ENABLE_SDP", "1") != "0":
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                logger.info("Enabled CUDA SDP backends (flash/mem_efficient/math)")
            except Exception as exc:  # pragma: no cover
                logger.debug("Failed to enable SDP backends: %s", exc)


def inference_context(device: str) -> Tuple[Any, Any]:
    """Autocast + inference_mode contexts keyed by device."""
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda"), torch.inference_mode()
    if device == "mps":
        return torch.amp.autocast(device_type="mps"), torch.inference_mode()
    return nullcontext(), torch.inference_mode()


def warmup_engine(engine: Any, max_new_tokens: int, prompt: str = "Warmup run") -> None:
    """Run a short generation to amortize cold-start overhead."""
    engine.run_generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stream_callback=lambda _: None,  # type: ignore[arg-type]
    )
