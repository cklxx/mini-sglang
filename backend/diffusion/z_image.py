"""Z-Image (diffusers) helpers and a small in-process runner.

The server uses this module to provide an OpenAI-compatible image generation endpoint:
`POST /v1/images/generations`.
"""

from __future__ import annotations

import inspect
import logging
import os
import secrets
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from utils.runtime import configure_torch

logger = logging.getLogger(__name__)


try:  # pragma: no cover - optional dependency
    from diffusers import GGUFQuantizationConfig, ZImagePipeline, ZImageTransformer2DModel
except Exception as exc:  # pragma: no cover
    GGUFQuantizationConfig = None  # type: ignore[assignment]
    ZImagePipeline = None  # type: ignore[assignment]
    ZImageTransformer2DModel = None  # type: ignore[assignment]
    _diffusers_import_error: Exception | None = exc
else:
    _diffusers_import_error = None


try:  # pragma: no cover - optional dependency
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception as exc:  # pragma: no cover
    hf_hub_download = None  # type: ignore[assignment]
    snapshot_download = None  # type: ignore[assignment]
    _hf_import_error: Exception | None = exc
else:
    _hf_import_error = None


DEFAULT_MODEL_ID = os.getenv("Z_IMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")

ASPECT_RATIOS: dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1088, 816),
    "3:4": (816, 1088),
}


def diffusion_available() -> bool:
    return ZImagePipeline is not None and snapshot_download is not None


def _require_diffusion() -> None:
    if ZImagePipeline is None:
        raise RuntimeError(f"diffusers is not available: {_diffusers_import_error}")
    if snapshot_download is None:
        raise RuntimeError(f"huggingface_hub is not available: {_hf_import_error}")


def pick_device(preferred: str = "auto") -> tuple[str, torch.dtype]:
    requested = preferred.lower()
    if requested != "auto":
        if requested == "mps" and torch.backends.mps.is_available():
            return "mps", torch.bfloat16
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda", torch.bfloat16
        if requested == "cpu":
            return "cpu", torch.float32
        logger.warning("Requested device %s unavailable; falling back to auto", preferred)

    if torch.backends.mps.is_available():
        return "mps", torch.bfloat16
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _resolve_model_dir(model_dir: str | None) -> Path:
    env_dir = os.getenv("Z_IMAGE_MODEL_DIR") or os.getenv("MODEL_LOCAL_DIR")
    if model_dir:
        return Path(model_dir).expanduser()
    if env_dir:
        return Path(env_dir).expanduser()
    cache_root = Path(os.getenv("Z_IMAGE_CACHE_DIR", Path.home() / ".cache" / "mini-sglang"))
    return cache_root.expanduser() / "z-image-turbo"


def ensure_model_available(model_id: str, model_dir: str | None) -> str:
    _require_diffusion()
    target_dir = _resolve_model_dir(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    if any(target_dir.iterdir()):
        logger.info("Using existing Z-Image model at %s", target_dir)
        return str(target_dir)

    logger.info("Downloading model %s to %s", model_id, target_dir)
    snapshot_download(  # type: ignore[misc]
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(target_dir)


def resolve_gguf_path(gguf: str, gguf_file: str | None, model_dir: str | None) -> str:
    _require_diffusion()
    candidate = Path(gguf).expanduser()
    if candidate.is_file():
        logger.info("Using local GGUF file at %s", candidate)
        return str(candidate)

    filename = gguf_file or "z_image_turbo-Q8_0.gguf"
    target_dir = _resolve_model_dir(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GGUF %s from %s into %s", filename, gguf, target_dir)
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub not installed; cannot download GGUF")
    return hf_hub_download(  # type: ignore[misc]
        repo_id=gguf,
        filename=filename,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )


def create_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def configure_attention(pipe: Any, backend: str) -> None:
    backend_map = {"sdpa": None, "flash2": "flash", "flash3": "_flash_3"}
    target = backend_map.get(backend, None)
    if not target:
        return

    try:
        pipe.transformer.set_attention_backend(target)
        logger.info("Using attention backend=%s", backend)
    except Exception as exc:  # pragma: no cover - best-effort configuration
        logger.warning("Could not enable %s attention (%s); using default SDPA", backend, exc)


def load_pipeline(
    *,
    model_path: str,
    device: str,
    dtype: torch.dtype,
    attention_backend: str,
    compile_transformer: bool,
    cpu_offload: bool,
    gguf_path: str | None = None,
    gguf_file: str | None = None,
    model_dir: str | None = None,
) -> Any:
    _require_diffusion()
    assert ZImagePipeline is not None
    load_kwargs: dict[str, object] = {"low_cpu_mem_usage": False}
    params = inspect.signature(ZImagePipeline.from_pretrained).parameters
    if "torch_dtype" in params:
        load_kwargs["torch_dtype"] = dtype
    elif "dtype" in params:
        load_kwargs["dtype"] = dtype

    transformer = None
    if gguf_path:
        assert GGUFQuantizationConfig is not None
        assert ZImageTransformer2DModel is not None
        resolved_gguf = resolve_gguf_path(gguf_path, gguf_file, model_dir)
        quant_config = GGUFQuantizationConfig(compute_dtype=dtype)
        transformer = ZImageTransformer2DModel.from_single_file(
            resolved_gguf, quantization_config=quant_config, dtype=dtype
        )
        logger.info("Loaded GGUF transformer from %s", resolved_gguf)

    pipe = ZImagePipeline.from_pretrained(model_path, transformer=transformer, **load_kwargs)
    if dtype != torch.float32 and hasattr(pipe, "vae"):
        pipe.vae.to(dtype=torch.float32)
        pipe.vae.config.force_upcast = True

    configure_attention(pipe, attention_backend)

    if cpu_offload and device.startswith("cuda"):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    if compile_transformer:
        try:
            pipe.transformer.compile()
            logger.info("Compiled DiT transformer for faster inference (first run may be slower).")
        except Exception as exc:  # pragma: no cover - optional optimization
            logger.warning("torch.compile failed (%s); continuing without compilation", exc)

    return pipe


def resolve_size(aspect: str | None, height: int, width: int) -> tuple[int, int]:
    if aspect:
        if aspect not in ASPECT_RATIOS:
            raise ValueError(f"unsupported aspect ratio: {aspect}")
        return ASPECT_RATIOS[aspect]
    return height, width


def parse_size(size: str | None) -> tuple[int, int]:
    if not size:
        return 1024, 1024
    if "x" not in size:
        raise ValueError("size must be like 1024x1024")
    w_str, h_str = size.lower().split("x", 1)
    width = int(w_str)
    height = int(h_str)
    if width <= 0 or height <= 0:
        raise ValueError("width/height must be > 0")
    return height, width


@dataclass(frozen=True)
class ZImageConfig:
    model_id: str
    model_dir: str | None
    device: str
    attention_backend: str
    compile_transformer: bool
    cpu_offload: bool
    gguf: str | None
    gguf_file: str | None


class ZImageRunner:
    def __init__(self, config: ZImageConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._pipe: Any | None = None
        self._device: str | None = None
        self._dtype: torch.dtype | None = None

    def _ensure_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe

        _require_diffusion()
        device, dtype = pick_device(self.config.device)
        configure_torch("cuda" if device.startswith("cuda") else device)
        model_path = ensure_model_available(self.config.model_id, self.config.model_dir)
        pipe = load_pipeline(
            model_path=model_path,
            device=device,
            dtype=dtype,
            attention_backend=self.config.attention_backend,
            compile_transformer=self.config.compile_transformer,
            cpu_offload=self.config.cpu_offload,
            gguf_path=self.config.gguf,
            gguf_file=self.config.gguf_file,
            model_dir=self.config.model_dir,
        )
        self._pipe = pipe
        self._device = device
        self._dtype = dtype
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

            images: list[Any] = []
            for idx in range(num_images):
                local_seed = int(seed) + idx if seed is not None else secrets.randbits(63)
                generator = create_generator(device, local_seed)
                out = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    height=height,
                    width=width,
                    generator=generator,
                )
                images.extend(getattr(out, "images", []))
            return images

