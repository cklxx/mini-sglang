"""Performance knobs and warmup helpers for mini-sglang."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import torch

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


def maybe_compile_model(model: Any, device: str, enabled: bool) -> Any:
    """Optionally wrap the model with torch.compile."""

    if not enabled:
        return model
    compile_mode = os.getenv("COMPILE_MODE", "reduce-overhead")
    try:
        compiled = torch.compile(model, mode=compile_mode)
        logger.info("Compiled model with torch.compile (mode=%s device=%s)", compile_mode, device)
        return compiled
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("torch.compile failed (%s); using eager model", exc)
        return model


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
