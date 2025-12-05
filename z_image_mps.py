"""Run Z-Image-Turbo locally with auto-download and simple CLI options."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
from diffusers import ZImagePipeline
from huggingface_hub import snapshot_download

from utils.runtime import configure_torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = os.getenv("Z_IMAGE_MODEL_ID", "Tongyi-MAI/Z-Image-Turbo")
DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu with intricate embroidery, holding a folding fan, "
    "soft outdoor night lighting, cinematic and detailed."
)

ASPECT_RATIOS: dict[str, Tuple[int, int]] = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1088, 816),
    "3:4": (816, 1088),
}


def pick_device(preferred: str = "auto") -> tuple[str, torch.dtype]:
    """Return best available device and matching dtype."""
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


def _resolve_model_dir(cli_dir: str | None) -> Path:
    env_dir = os.getenv("Z_IMAGE_MODEL_DIR") or os.getenv("MODEL_LOCAL_DIR")
    if cli_dir:
        return Path(cli_dir).expanduser()
    if env_dir:
        return Path(env_dir).expanduser()
    cache_root = Path(
        os.getenv("Z_IMAGE_CACHE_DIR", Path.home() / ".cache" / "mini-sglang")
    ).expanduser()
    return cache_root / "z-image-turbo"


def ensure_model_available(model_id: str, model_dir: str | None) -> str:
    """Ensure the Z-Image model is present locally and return its path."""
    target_dir = _resolve_model_dir(model_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    has_contents = any(target_dir.iterdir())
    if has_contents:
        logger.info("Using existing Z-Image model at %s", target_dir)
        return str(target_dir)

    logger.info("Downloading model %s to %s", model_id, target_dir)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    return str(target_dir)


def create_generator(device: str, seed: int) -> torch.Generator:
    generator_device = "cpu" if device == "mps" else device
    return torch.Generator(device=generator_device).manual_seed(seed)


def configure_attention(pipe: ZImagePipeline, backend: str) -> None:
    backend_map = {
        "sdpa": None,
        "flash2": "flash",
        "flash3": "_flash_3",
    }
    target = backend_map.get(backend, None)
    if not target:
        return

    try:
        pipe.transformer.set_attention_backend(target)
        logger.info("Using attention backend=%s", backend)
    except Exception as exc:  # pragma: no cover - best-effort configuration
        logger.warning("Could not enable %s attention (%s); using default SDPA", backend, exc)


def load_pipeline(
    model_path: str,
    device: str,
    dtype: torch.dtype,
    attention_backend: str,
    compile_transformer: bool,
    cpu_offload: bool,
) -> ZImagePipeline:
    load_kwargs: dict[str, object] = {"low_cpu_mem_usage": False}
    params = inspect.signature(ZImagePipeline.from_pretrained).parameters
    if "torch_dtype" in params:
        load_kwargs["torch_dtype"] = dtype
    elif "dtype" in params:
        load_kwargs["dtype"] = dtype

    pipe = ZImagePipeline.from_pretrained(model_path, **load_kwargs)

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
            logger.info(
                "Compiled DiT transformer for faster inference (first run may be slower)."
            )
        except Exception as exc:  # pragma: no cover - optional optimization
            logger.warning("torch.compile failed (%s); continuing without compilation", exc)

    return pipe


def resolve_size(aspect: str | None, height: int, width: int) -> tuple[int, int]:
    if aspect:
        return ASPECT_RATIOS[aspect]
    return height, width


def build_output_path(
    output: str | None, outdir: str | None, timestamp: str, index: int, total: int
) -> Path:
    suffix = f"-{index}" if total > 1 else ""
    if output:
        target = Path(output).expanduser()
        if output.endswith(os.sep) or target.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            return target / f"z-image-{timestamp}{suffix}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        stem = target.stem or "z-image"
        ext = target.suffix or ".png"
        return target.with_name(f"{stem}{suffix}{ext}")

    root = Path(outdir or "output").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root / f"z-image-{timestamp}{suffix}.png"


def run_generation(args: argparse.Namespace) -> None:
    device, dtype = pick_device(args.device)
    configure_torch(device)
    logger.info("Using device=%s dtype=%s", device, dtype)

    model_path = ensure_model_available(args.model, args.model_dir)
    pipe = load_pipeline(
        model_path=model_path,
        device=device,
        dtype=dtype,
        attention_backend=args.attention_backend,
        compile_transformer=args.compile,
        cpu_offload=args.cpu_offload,
    )

    height, width = resolve_size(args.aspect, args.height, args.width)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    num_images = max(1, args.num_images)
    for image_index in range(num_images):
        if args.seed is not None:
            seed = int(args.seed) + image_index
        else:
            seed = secrets.randbits(63)
        generator = create_generator(device, seed)

        logger.info(
            "[%d/%d] prompt=%r steps=%d guidance=%.2f seed=%d size=%dx%d",
            image_index + 1,
            num_images,
            args.prompt,
            args.steps,
            args.guidance_scale,
            seed,
            width,
            height,
        )

        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )

        output_path = build_output_path(
            output=args.output,
            outdir=args.outdir,
            timestamp=timestamp,
            index=image_index + 1,
            total=num_images,
        )
        result.images[0].save(output_path)
        logger.info("Saved image to %s", output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate images with Z-Image-Turbo (Apple MPS, CUDA, or CPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-p", "--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt.")
    parser.add_argument("--negative-prompt", type=str, default=None, help="Negative prompt text.")
    parser.add_argument("-s", "--steps", type=int, default=9, help="Number of inference steps.")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale (Turbo checkpoints expect 0.0).",
    )
    parser.add_argument("--height", type=int, default=1024, help="Image height (px).")
    parser.add_argument("--width", type=int, default=1024, help="Image width (px).")
    parser.add_argument(
        "--aspect",
        choices=sorted(ASPECT_RATIOS.keys()),
        default=None,
        help="Aspect ratio presets; overrides height/width when set.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility.")
    parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Generate multiple images (seeds increment when a base seed is provided).",
    )
    parser.add_argument("-o", "--output", type=str, default=None, help="Output file or directory.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Directory for outputs (ignored when --output is a file path).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Force a device or let the script pick automatically.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flash2", "flash3"],
        default="sdpa",
        help="Attention backend for the DiT transformer.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Try torch.compile on the transformer (best on CUDA).",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Enable CPU offload (CUDA only) to reduce VRAM usage.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help="Model repo or local path for the Z-Image pipeline.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Optional local directory for the model; defaults to Z_IMAGE_MODEL_DIR or cache.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
