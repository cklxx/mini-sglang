"""Backend selection helpers for torch vs MLX."""

from __future__ import annotations

import logging
import os
from typing import Any

from backend.model_backend import ModelBackend

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
    """Resolve backend implementation ('torch' | 'mlx')."""

    impl = (os.getenv("BACKEND_IMPL") or "auto").lower()
    if impl not in {"auto", "torch", "mlx"}:
        logger.warning("Unrecognized BACKEND_IMPL=%s; defaulting to auto", impl)
        impl = "auto"
    if impl == "torch":
        return "torch"
    if impl == "mlx":
        return "mlx"
    if device == "mps" and _mlx_available():
        return "mlx"
    return "torch"


def backend_label(backend: Any) -> str:
    """Human-readable backend name for logging."""

    module = getattr(backend.__class__, "__module__", "")
    if module.startswith("backend.mlx_backend"):
        return "mlx"
    return "torch"


def create_backend(
    model_name: str, device: str, compile_model: bool = False
) -> Any:
    """Instantiate either the torch or MLX backend based on env + device."""

    impl = resolve_backend_impl(device)
    if impl == "mlx":
        from backend.mlx_backend import MlxBackend
        return MlxBackend(model_name=model_name, device=device)

    return ModelBackend(model_name=model_name, device=device, compile_model=compile_model)
