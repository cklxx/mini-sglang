"""Backend selection helpers for torch/sgl_kernel/MLX/HF baseline."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def _mlx_available() -> bool:
    try:
        import mlx  # noqa: F401
        import mlx_lm  # noqa: F401
    except Exception as exc:
        logger.debug("MLX backend unavailable: %s", exc)
        return False
    return True


def resolve_backend_impl(device: str) -> str:
    """Resolve backend implementation ('torch' | 'sgl' | 'mlx' | 'hf')."""

    impl = (os.getenv("BACKEND_IMPL") or "auto").lower()
    if impl not in {"auto", "torch", "sgl", "mlx", "hf"}:
        logger.warning("Unrecognized BACKEND_IMPL=%s; defaulting to auto", impl)
        impl = "auto"
    if impl in {"torch", "hf"}:
        return impl
    if impl == "mlx":
        return "mlx"
    if impl == "sgl":
        return "sgl"
    if device == "mps" and _mlx_available():
        return "mlx"
    if device.startswith("cuda"):
        return "sgl"
    return "torch"


def backend_label(backend: Any) -> str:
    """Human-readable backend name for logging."""

    module = getattr(backend.__class__, "__module__", "")
    if module.startswith("backend.mlx"):
        return "mlx"
    if module.startswith("backend.sglang"):
        return "sgl"
    return "torch"


def create_backend(model_name: str, device: str, compile_model: bool = False, **_: Any) -> Any:
    """Instantiate selected backend based on env + device."""

    impl = resolve_backend_impl(device)
    if impl == "mlx":
        from backend.mlx import MlxBackend

        return MlxBackend(model_name=model_name, device=device)
    if impl == "sgl":
        try:
            from backend.sglang import SglKernelQwenBackend

            return SglKernelQwenBackend(model_name=model_name, device=device)
        except Exception as exc:
            logger.warning(
                "sgl_kernel backend unavailable (%s); falling back to HF/torch backend", exc
            )

    # torch and hf fall back to the simplified torch backend
    from backend.hf import ModelBackend

    if compile_model:
        logger.info("compile_model flag is ignored for the HF/torch backend")
    return ModelBackend(model_name=model_name, device=device)
