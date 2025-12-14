"""MLX-powered VLM runner (best-effort, optional dependency).

This integrates with third-party MLX VLM packages when available. It is intentionally implemented
as a small adapter layer so the FastAPI server can pick a VLM backend based on device and env.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import threading
import time
from typing import Any, Callable, Iterable, Optional, Tuple, cast

logger = logging.getLogger(__name__)


def _find_callable(candidates: list[tuple[str, str]]) -> Callable[..., Any]:
    last_exc: Exception | None = None
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            last_exc = exc
            continue
        fn = getattr(module, attr, None)
        if callable(fn):
            return fn
    detail = f" (last_error={last_exc})" if last_exc else ""
    raise ImportError(f"Could not locate required MLX VLM callable{detail}")


def _filtered_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _call_best_effort(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    try:
        return fn(*args, **_filtered_kwargs(fn, kwargs))
    except TypeError:
        # Retry without kwargs if the callee is strict and our guess was wrong.
        return fn(*args)


class MlxVLMRunner:
    """Best-effort MLX VLM runner with a streaming-compatible API.

    This runner requires a separate MLX VLM package (not part of mlx-lm). When unavailable, the
    FastAPI server should fall back to the HF VLM runner.
    """

    def __init__(self, model_name: str, device: str = "mps") -> None:
        if device != "mps":
            raise ValueError("MlxVLMRunner only supports device='mps'")

        # Ensure MLX core exists.
        try:
            import mlx.core as mx  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "MlxVLMRunner requires the `mlx` package. Install with `pip install mlx`."
            ) from exc

        # Try to locate load/generate entrypoints from a third-party mlx_vlm package.
        self._load_fn = _find_callable(
            [
                ("mlx_vlm", "load"),
                ("mlx_vlm.utils", "load"),
                ("mlx_vlm.utils", "load_model"),
                ("mlx_vlm", "load_model"),
            ]
        )
        self._stream_fn = None
        try:
            self._stream_fn = _find_callable(
                [
                    ("mlx_vlm", "stream_generate"),
                    ("mlx_vlm.generate", "stream_generate"),
                    ("mlx_vlm", "generate_stream"),
                ]
            )
        except Exception:
            self._stream_fn = None
        self._generate_fn = None
        try:
            self._generate_fn = _find_callable(
                [
                    ("mlx_vlm", "generate"),
                    ("mlx_vlm.generate", "generate"),
                ]
            )
        except Exception:
            self._generate_fn = None

        self.model_name = model_name
        self.device = device
        self._lock = threading.Lock()

        trust_remote_code = os.getenv("MLX_TRUST_REMOTE_CODE", "0") != "0"
        loaded = _call_best_effort(
            self._load_fn, (model_name,), {"trust_remote_code": trust_remote_code}
        )
        if isinstance(loaded, tuple):
            self.model = loaded[0]
            self.processor = loaded[1] if len(loaded) > 1 else None
            self.tokenizer = loaded[2] if len(loaded) > 2 else self.processor
        else:
            self.model = loaded
            self.processor = None
            self.tokenizer = None

    def tokenize(self, text: str) -> list[int]:
        tok = self.tokenizer
        if tok is None:
            return []
        encode = getattr(tok, "encode", None)
        if callable(encode):
            try:
                return list(encode(text, add_special_tokens=False))
            except Exception:
                try:
                    return list(encode(text))
                except Exception:
                    return []
        return []

    def generate_streaming(
        self,
        *,
        prompt: str,
        images: list[Any],
        max_new_tokens: int,
        log_stride: int = 32,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, float]:
        if not images:
            raise ValueError("MlxVLMRunner requires at least one image input")

        with self._lock:
            start = time.perf_counter()
            chunks: list[str] = []

            if self._stream_fn is not None:
                stream_iter = _call_best_effort(
                    self._stream_fn,
                    (),
                    {
                        "model": self.model,
                        "processor": self.processor,
                        "prompt": prompt,
                        "images": images,
                        "max_new_tokens": max_new_tokens,
                    },
                )
                for idx, item in enumerate(cast(Iterable[Any], stream_iter), 1):
                    delta: str | None = None
                    if isinstance(item, str):
                        delta = item
                    elif isinstance(item, tuple) and item and isinstance(item[0], str):
                        delta = item[0]
                    elif isinstance(item, dict):
                        maybe = item.get("text") or item.get("delta")
                        if isinstance(maybe, str):
                            delta = maybe
                    if not delta:
                        continue
                    if stream_callback:
                        stream_callback(delta)
                    chunks.append(delta)
                    if idx == 1 or idx % log_stride == 0:
                        logger.info("[mlx-vlm] chunk %03d: %r", idx, delta)
            else:
                if self._generate_fn is None:
                    raise ImportError(
                        "mlx_vlm does not expose generate/stream_generate; "
                        "install a compatible mlx-vlm package."
                    )
                out = _call_best_effort(
                    self._generate_fn,
                    (),
                    {
                        "model": self.model,
                        "processor": self.processor,
                        "prompt": prompt,
                        "images": images,
                        "max_new_tokens": max_new_tokens,
                    },
                )
                text = out if isinstance(out, str) else str(out)
                if stream_callback:
                    stream_callback(text)
                chunks.append(text)

            duration = time.perf_counter() - start
            return "".join(chunks), duration
