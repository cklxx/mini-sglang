"""Generic diffusers text-to-image runner (optional dependency)."""

from __future__ import annotations

import inspect
import logging
import secrets
import threading
from dataclasses import dataclass
from typing import Any, Optional

import torch

from backend.diffusion.z_image import create_generator, ensure_model_available, pick_device
from utils.runtime import configure_torch

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    from diffusers import DiffusionPipeline
except Exception as exc:  # pragma: no cover
    DiffusionPipeline = None  # type: ignore[assignment]
    _diffusers_import_error: Exception | None = exc
else:
    _diffusers_import_error = None


def _require_diffusers() -> None:
    if DiffusionPipeline is None:
        raise RuntimeError(f"diffusers is not available: {_diffusers_import_error}")


def _filtered_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except Exception:
        return kwargs
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return kwargs
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


@dataclass(frozen=True)
class DiffusersImageConfig:
    model_id: str
    model_dir: str | None
    device: str
    cpu_offload: bool


class DiffusersImageRunner:
    def __init__(self, config: DiffusersImageConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._pipe: Any | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        _require_diffusers()
        assert DiffusionPipeline is not None

        device, dtype = pick_device(self.config.device)
        configure_torch("cuda" if device.startswith("cuda") else device)

        model_path = ensure_model_available(self.config.model_id, self.config.model_dir)
        pipe = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype)

        if (
            self.config.cpu_offload
            and device.startswith("cuda")
            and hasattr(pipe, "enable_model_cpu_offload")
        ):
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        self._pipe = pipe
        self._device = device
        self._dtype = dtype
        logger.info(
            "Loaded diffusers image pipeline | model=%s device=%s", self.config.model_id, device
        )
        return pipe

    def generate(
        self,
        *,
        prompt: str,
        num_images: int,
        steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        seed: Optional[int],
    ) -> list[Any]:
        if num_images <= 0:
            raise ValueError("num_images must be >= 1")
        if steps <= 0:
            raise ValueError("steps must be >= 1")

        with self._lock:
            pipe = self._ensure_pipe()
            device = self._device or "cpu"
            call = getattr(pipe, "__call__", None)
            if call is None:
                raise RuntimeError("diffusers pipeline is not callable")

            images: list[Any] = []
            for idx in range(num_images):
                local_seed = int(seed) + idx if seed is not None else secrets.randbits(63)
                generator = create_generator(device, local_seed)
                kwargs = _filtered_kwargs(
                    call,
                    {
                        "prompt": prompt,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width,
                        "generator": generator,
                    },
                )
                out = call(**kwargs)
                batch = getattr(out, "images", None) or getattr(out, "image", None) or out
                if isinstance(batch, list):
                    images.extend(batch)
                else:
                    images.append(batch)
            return images
