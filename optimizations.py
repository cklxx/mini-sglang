"""Performance knobs and warmup helpers for mini-sglang."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch
from contextlib import nullcontext

logger = logging.getLogger(__name__)


def configure_torch(device: str) -> None:
    """Set global torch knobs for better perf on GPU/MPS."""

    # Prefer faster matmul when safe.
    precision = os.getenv("MATMUL_PRECISION")
    if precision:
        torch.set_float32_matmul_precision(precision)
    elif device == "cuda":
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        if os.getenv("ENABLE_SDP", "1") != "0":
            # Enable fused/flash attention when available (torch 2.1+).
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                torch.backends.cuda.enable_math_sdp(True)
                logger.info("Enabled CUDA SDP backends (flash/mem_efficient/math)")
            except Exception as exc:  # pragma: no cover - best-effort
                logger.debug("Failed to enable SDP backends: %s", exc)


def maybe_compile_model(model: Any, device: str, enabled: bool) -> Any:
    """Optionally wrap the model with torch.compile."""

    if not enabled:
        return model

    # torch.compile is most stable on CUDA; avoid on MPS/CPU unless explicitly forced.
    if device != "cuda" and os.getenv("FORCE_COMPILE_ON_NONCUDA") != "1":
        logger.info("Skipping torch.compile on non-CUDA device=%s", device)
        return model

    compile_mode = os.getenv("COMPILE_MODE", "reduce-overhead")
    try:
        compiled = torch.compile(model, mode=compile_mode)
        logger.info("Compiled model with torch.compile (mode=%s device=%s)", compile_mode, device)
        return compiled
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("torch.compile failed (%s); using eager model", exc)
        return model


def inference_context(device: str):
    """Lightweight context for inference-only sections."""

    # Autocast only on CUDA/MPS; CPU autocast is limited.
    if device == "cuda":
        return torch.amp.autocast(device_type="cuda"), torch.inference_mode()
    if device == "mps":
        return torch.amp.autocast(device_type="mps"), torch.inference_mode()
    return nullcontext(), torch.inference_mode()


def warmup_engine(engine: Any, max_new_tokens: int, prompt: str = "Warmup run") -> None:
    """Run a short generation to amortize cold-start overhead."""

    start = time.perf_counter()
    engine.run_generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        stream_callback=lambda _: None,  # discard warmup output
    )
    duration = time.perf_counter() - start
    logger.info("Warmup complete | tokens=%d duration=%.3fs", max_new_tokens, duration)
