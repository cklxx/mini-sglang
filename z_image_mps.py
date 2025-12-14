"""Run Z-Image-Turbo locally with auto-download and simple CLI options."""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import secrets
from datetime import datetime
from pathlib import Path
from time import perf_counter

import torch

from backend.diffusion.z_image import (
    ASPECT_RATIOS,
    DEFAULT_MODEL_ID,
    create_generator,
    ensure_model_available,
    load_pipeline,
    pick_device,
    resolve_size,
)
from utils.runtime import configure_torch

logger = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "Young Chinese woman in red Hanfu with intricate embroidery, holding a folding fan, "
    "soft outdoor night lighting, cinematic and detailed."
)


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
        gguf_path=args.gguf,
        gguf_file=args.gguf_file,
        model_dir=args.model_dir,
    )

    supports_callback = False
    try:
        supports_callback = "callback" in inspect.signature(pipe.__call__).parameters
    except Exception:  # pragma: no cover - defensive
        supports_callback = False

    height, width = resolve_size(args.aspect, args.height, args.width)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    num_images = max(1, args.num_images)
    overall_start = perf_counter()
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

        callback_steps = max(1, args.steps // 10) if args.steps > 0 else 1
        last_step_log = -1

        def progress_callback(step: int, timestep: int, latents: object | None = None) -> None:
            nonlocal last_step_log
            if step == last_step_log:
                return
            if step in {0, args.steps - 1} or step - last_step_log >= callback_steps:
                pct = int(((step + 1) / max(1, args.steps)) * 100)
                logger.info("  progress: step %d/%d (%d%%)", step + 1, args.steps, pct)
                last_step_log = step

        generation_kwargs: dict[str, object] = {}
        if supports_callback:
            generation_kwargs["callback"] = progress_callback
            generation_kwargs["callback_steps"] = callback_steps

        image_start = perf_counter()
        with torch.inference_mode():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=height,
                width=width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                **generation_kwargs,
            )

        output_path = build_output_path(
            output=args.output,
            outdir=args.outdir,
            timestamp=timestamp,
            index=image_index + 1,
            total=num_images,
        )
        result.images[0].save(output_path)
        image_elapsed = perf_counter() - image_start
        logger.info("Saved image to %s (%.2fs)", output_path, image_elapsed)

    total_elapsed = perf_counter() - overall_start
    logger.info("Finished %d image(s) in %.2fs", num_images, total_elapsed)


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
        "--gguf",
        type=str,
        default=None,
        help=(
            "Optional GGUF transformer file (path or repo id such as jayn7/Z-Image-Turbo-GGUF). "
            "When set, only the DiT transformer loads from GGUF and the rest of the pipeline "
            "comes from --model."
        ),
    )
    parser.add_argument(
        "--gguf-file",
        type=str,
        default="z_image_turbo-Q8_0.gguf",
        help="GGUF filename to fetch when --gguf points to a repo id.",
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
