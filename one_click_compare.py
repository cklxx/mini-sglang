"""One-click comparison between mini-sglang streaming and vanilla HF generate.

This script is the **all-in-one, teaching-friendly** entrypoint:
- Auto-installs dependencies via `uv pip` when available (falls back to `pip`).
- Defaults to the tiniest public text model so the first run is quick.
- Prints readable logs that map each step back to sglang's API / Engine /
  Backend layering.

It simplifies the sglang experience into a single command a new learner can
run to see both streaming (prefill + decode) and traditional batched
generation side by side.
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Tuple
import platform

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_TINY_MODEL = "sshleifer/tiny-gpt2"
REQUIREMENTS_PATH = Path(__file__).resolve().parent / "requirements.txt"


# ---------------------------------------------------------------------------
# Bootstrap helpers (kept stdlib-only so they work before deps are installed)
# ---------------------------------------------------------------------------

def ensure_dependencies() -> None:
    """Install dependencies if missing, preferring `uv pip` for speed.

    This keeps the script truly one-click: it will try to install torch,
    transformers, fastapi, and uvicorn if they are absent. When uv is not
    present, it will optionally auto-install uv first so that the remaining
    install uses the faster resolver. Set ``AUTO_INSTALL_UV=0`` to skip uv
    bootstrapping.
    """

    needed = []
    for pkg in ("torch", "transformers", "fastapi", "uvicorn"):
        if importlib.util.find_spec(pkg) is None:
            needed.append(pkg)
    if not needed:
        return

    uv_bin = os.environ.get("UV_BIN") or shutil.which("uv")  # type: ignore[name-defined]
    if uv_bin is None and os.environ.get("AUTO_INSTALL_UV", "1") != "0":
        try:
            print("[bootstrap] uv not found, installing with pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
            uv_bin = shutil.which("uv")
        except subprocess.CalledProcessError:
            print("[bootstrap] Failed to install uv, falling back to pip.")

    cmd: list[str]
    runner: str

    index_url = os.environ.get("TORCH_INDEX_URL")
    # Default to CPU wheels on Linux so CPU-only machines avoid CUDA downloads.
    if (
        index_url is None
        and platform.system() == "Linux"
        and os.environ.get("ALLOW_CUDA_TORCH") is None
    ):
        index_url = "https://download.pytorch.org/whl/cpu"

    if uv_bin:
        cmd = [uv_bin, "pip", "install"]
        runner = "uv"
    else:
        cmd = [sys.executable, "-m", "pip", "install"]
        runner = "pip"

    if index_url:
        cmd.extend(["--index-url", index_url])
    cmd.extend(["-r", str(REQUIREMENTS_PATH)])

    print(
        f"[bootstrap] Installing dependencies with {runner}"
        + (f" (index_url={index_url})" if index_url else "")
        + "..."
    )
    subprocess.check_call(cmd)


# shutil is intentionally imported lazily after defining ensure_dependencies to
# keep the top of the file free of non-stdlib imports that might not exist yet.
import shutil  # noqa: E402  # isort:skip


# ---------------------------------------------------------------------------
# Lazy imports after deps are ensured
# ---------------------------------------------------------------------------

def load_components():
    """Import project components only after deps exist."""

    from backend.model_backend import ModelBackend
    from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
    from engine.engine import SGLangMiniEngine

    return ModelBackend, SGLangMiniEngine, MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-click streaming vs traditional generate comparison",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Hello from mini-sglang",
        help="Prompt to feed the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        dest="max_new_tokens",
        default=None,
        help="Token budget for both modes (defaults to project setting)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override model name. Defaults to MODEL_NAME env or a tiny text model"
            " for fastest first run."
        ),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity (DEBUG/INFO/WARNING)",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip dependency auto-install (for pre-provisioned environments)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core comparison helpers (typed loosely to avoid early imports)
# ---------------------------------------------------------------------------

def build_backend(model_name: str, ModelBackend: Any, get_device: Any) -> Any:
    device = get_device()
    logging.info(
        "Preparing backend with model=%s on device=%s (auto-download on first run)",
        model_name,
        device,
    )
    return ModelBackend(model_name=model_name, device=device)


def run_streaming(
    engine: Any,
    prompt: str,
    max_new_tokens: Optional[int],
) -> Tuple[str, float]:
    chunks: list[str] = []
    start = time.perf_counter()

    def stream_callback(text_delta: str) -> None:
        chunks.append(text_delta)
        logging.info("[stream] chunk %03d: %r", len(chunks), text_delta)

    final_text = engine.run_generate(
        prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=stream_callback
    )
    duration = time.perf_counter() - start
    return final_text, duration


def run_traditional(
    backend: Any, prompt: str, max_new_tokens: int
) -> Tuple[str, float]:
    prompt_ids = backend.tokenize(prompt)
    start = time.perf_counter()
    generated_ids = backend.generate_greedy(
        prompt_ids=prompt_ids, max_new_tokens=max_new_tokens
    )
    text = backend.decode_tokens(generated_ids)
    duration = time.perf_counter() - start
    return text, duration


def summarize(
    *,
    mode: str,
    text: str,
    duration: float,
    backend: Any,
) -> None:
    tokens = len(backend.tokenize(text))
    logging.info(
        "%s summary: tokens=%d duration=%.3fs throughput=%.2f tok/s",
        mode,
        tokens,
        duration,
        tokens / duration if duration > 0 else 0.0,
    )
    print(f"\n{mode} output preview: {text[:120]!r}\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    if not args.no_bootstrap:
        ensure_dependencies()

    ModelBackend, SGLangMiniEngine, MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device = (
        load_components()
    )

    model_name = (
        args.model
        or os.environ.get("MODEL_NAME")
        or MODEL_NAME
        or DEFAULT_TINY_MODEL
    )
    if model_name == MODEL_NAME and MODEL_NAME == "gpt2":
        # Prefer the tiny model for a faster teaching demo when MODEL_NAME isn't set.
        model_name = DEFAULT_TINY_MODEL
    os.environ.setdefault("MODEL_NAME", model_name)

    backend = build_backend(model_name, ModelBackend, get_device)
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    token_budget = args.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
    logging.info(
        "Starting comparison with max_new_tokens=%d (prompt=%r)",
        token_budget,
        args.prompt,
    )

    logging.info("=== Streaming via mini-sglang (prefill + decode) ===")
    stream_text, stream_duration = run_streaming(
        engine=engine, prompt=args.prompt, max_new_tokens=token_budget
    )
    summarize(mode="Streaming", text=stream_text, duration=stream_duration, backend=backend)

    logging.info("=== Traditional single-call generate() ===")
    greedy_text, greedy_duration = run_traditional(
        backend=backend, prompt=args.prompt, max_new_tokens=token_budget
    )
    summarize(
        mode="Traditional generate", text=greedy_text, duration=greedy_duration, backend=backend
    )

    print("Done. Use the logs above to trace each step.")


if __name__ == "__main__":
    main()
