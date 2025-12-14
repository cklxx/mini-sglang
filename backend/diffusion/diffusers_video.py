"""Generic diffusers video generation runner (optional dependency).

This module targets pipelines that return a list of frames (PIL images) via an output attribute
like `.frames` or `.videos`. The FastAPI server can expose this through an OpenAI-like endpoint.
"""

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


def _extract_frames(output: Any) -> list[Any]:
    if output is None:
        return []

    if isinstance(output, dict):
        if "frames" in output:
            output = output["frames"]
        elif "videos" in output:
            output = output["videos"]

    frames = getattr(output, "frames", None)
    if frames is not None:
        # StableVideoDiffusion commonly returns frames as: List[List[PIL.Image]]
        if isinstance(frames, list) and frames and isinstance(frames[0], list):
            return list(frames[0])
        if isinstance(frames, list):
            return list(frames)

    videos = getattr(output, "videos", None)
    if videos is not None:
        # As a fallback, return raw arrays (encoding handled by the API layer).
        if isinstance(videos, list):
            return videos
        return [videos]

    if isinstance(output, list):
        return output
    return []


@dataclass(frozen=True)
class DiffusersVideoConfig:
    model_id: str
    model_dir: str | None
    device: str
    cpu_offload: bool


class DiffusersVideoRunner:
    def __init__(self, config: DiffusersVideoConfig) -> None:
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
            "Loaded diffusers video pipeline | model=%s device=%s", self.config.model_id, device
        )
        return pipe

    def generate(
        self,
        *,
        prompt: str | None,
        image: Any | None,
        num_videos: int,
        steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        num_frames: int,
        fps: int,
        seed: Optional[int],
    ) -> list[list[Any]]:
        if num_videos <= 0:
            raise ValueError("num_videos must be >= 1")
        if steps <= 0:
            raise ValueError("steps must be >= 1")
        if num_frames <= 0:
            raise ValueError("num_frames must be >= 1")
        if fps <= 0:
            raise ValueError("fps must be >= 1")

        with self._lock:
            pipe = self._ensure_pipe()
            device = self._device or "cpu"
            call = getattr(pipe, "__call__", None)
            if call is None:
                raise RuntimeError("diffusers pipeline is not callable")

            videos: list[list[Any]] = []
            for idx in range(num_videos):
                local_seed = int(seed) + idx if seed is not None else secrets.randbits(63)
                generator = create_generator(device, local_seed)
                kwargs = _filtered_kwargs(
                    call,
                    {
                        "prompt": prompt,
                        "image": image,
                        "num_inference_steps": steps,
                        "guidance_scale": guidance_scale,
                        "height": height,
                        "width": width,
                        "num_frames": num_frames,
                        "fps": fps,
                        "generator": generator,
                    },
                )
                out = call(**kwargs)
                frames = _extract_frames(out)
                if not frames:
                    raise RuntimeError("video pipeline returned no frames")
                videos.append(frames)
            return videos
