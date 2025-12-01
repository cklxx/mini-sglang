"""One-click comparison between mini-sglang streaming and vanilla HF generate.

This script is the **all-in-one, teaching-friendly** entrypoint:
- Auto-installs dependencies via `uv pip` when available (falls back to `pip`).
- Defaults to a modern, lightweight instruct model so the first run has decent
  quality without a large GPU.
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
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default to a recent, lightweight instruct model so quality is decent without
# requiring a large GPU.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
BENCHMARK_SHORT_PROMPT = (
    "In four bullet points, explain how streaming token generation reduces perceived"
    " latency and improves user trust in a chat UI."
)
BENCHMARK_LONG_PROMPT = """You are preparing a system design update for a latency-sensitive LLM chat service.
Write a concise summary (<=120 words) plus two risks and two next steps.
Context:
- We serve 50k RPM with a p95 SLA of 1.5s for user-visible tokens.
- We are replacing one-shot generate with streaming prefill+decode to cut TTFB.
- Clients show typing indicators and rely on early tokens to keep users engaged.
- Infra: Apple M-series laptops for demos, CUDA for staging and load tests.
- Batching helped throughput but raised TTFB under bursty traffic.
- Metrics: TTFB dropped from 900ms to 150ms in A/B when streaming; tail latency improved modestly.
- Content quality must stay consistent between streaming and batch output."""
BENCHMARK_TURNS = [
    "Draft a two-sentence update about migrating from batched generate to streaming decode for an LLM chat service.",
    "Rewrite the update as three crisp numbered bullets with a latency metric placeholder.",
]
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
    for pkg in ("torch", "transformers", "fastapi", "uvicorn", "sentencepiece"):
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
        nargs="*",
        help=(
            "Prompt(s) to feed the model. Provide multiple values with --multi-turn "
            "to simulate a chat benchmark. If omitted, the preset benchmark suite "
            "will run."
        ),
    )
    parser.add_argument(
        "--multi-turn",
        action="store_true",
        help="Treat the prompts as sequential user turns for a chat benchmark",
    )
    parser.add_argument(
        "--benchmark-suite",
        action="store_true",
        help="Run a preset short + long benchmark and print a throughput comparison table",
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
            "Override model name. Defaults to MODEL_NAME env or a modern small instruct model."
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


def build_chat_prompt(history: list[tuple[str, str]], user_turn: str) -> str:
    """Format chat history into a prompt."""

    segments: list[str] = []
    for prev_user, prev_bot in history:
        segments.append(f"User: {prev_user}\nAssistant: {prev_bot}")
    segments.append(f"User: {user_turn}\nAssistant:")
    return "\n".join(segments)


def run_streaming_chat(
    *, engine: Any, user_turns: Sequence[str], max_new_tokens: Optional[int]
) -> Tuple[list[tuple[str, str]], float]:
    history: list[tuple[str, str]] = []
    total_duration = 0.0

    for turn_index, user_turn in enumerate(user_turns, 1):
        prompt = build_chat_prompt(history, user_turn)
        chunks: list[str] = []
        start = time.perf_counter()

        def stream_callback(text_delta: str) -> None:
            chunks.append(text_delta)
            logging.info(
                "[stream][turn %d] chunk %03d: %r",
                turn_index,
                len(chunks),
                text_delta,
            )

        response = engine.run_generate(
            prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=stream_callback
        )
        total_duration += time.perf_counter() - start
        history.append((user_turn, response))

    return history, total_duration


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


def run_traditional_chat(
    *, backend: Any, user_turns: Sequence[str], max_new_tokens: int
) -> Tuple[list[tuple[str, str]], float]:
    history: list[tuple[str, str]] = []
    total_duration = 0.0

    for user_turn in user_turns:
        prompt = build_chat_prompt(history, user_turn)
        prompt_ids = backend.tokenize(prompt)
        start = time.perf_counter()
        generated_ids = backend.generate_greedy(
            prompt_ids=prompt_ids, max_new_tokens=max_new_tokens
        )
        response = backend.decode_tokens(generated_ids)
        total_duration += time.perf_counter() - start
        history.append((user_turn, response))

    return history, total_duration


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


def summarize_chat(
    *, mode: str, history: list[tuple[str, str]], duration: float, backend: Any
) -> None:
    responses = [bot for _, bot in history]
    combined = "\n".join(responses)
    tokens = len(backend.tokenize(combined)) if responses else 0
    logging.info(
        "%s chat summary: turns=%d tokens=%d duration=%.3fs throughput=%.2f tok/s",
        mode,
        len(history),
        tokens,
        duration,
        tokens / duration if duration > 0 else 0.0,
    )
    last_reply = responses[-1] if responses else ""
    print(f"\n{mode} last reply preview: {last_reply[:120]!r}\n")


def run_benchmark_suite(
    *, engine: Any, backend: Any, max_new_tokens: int
) -> None:
    """Run short + long prompts (and a refinement turn) and print a speed table."""

    scenarios = [
        {
            "label": "Short instruction",
            "multi_turn": False,
            "payload": BENCHMARK_SHORT_PROMPT,
        },
        {
            "label": "Long context summarization",
            "multi_turn": False,
            "payload": BENCHMARK_LONG_PROMPT,
        },
        {
            "label": "Two-turn refinement",
            "multi_turn": True,
            "payload": BENCHMARK_TURNS,
        },
    ]

    rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        label = scenario["label"]
        if scenario["multi_turn"]:
            user_turns = scenario["payload"]
            logging.info("=== %s | Streaming (chat) ===", label)
            stream_history, stream_duration = run_streaming_chat(
                engine=engine, user_turns=user_turns, max_new_tokens=max_new_tokens
            )
            summarize_chat(
                mode=f"{label} | Streaming",
                history=stream_history,
                duration=stream_duration,
                backend=backend,
            )
            stream_tokens = len(backend.tokenize(" ".join(r for _, r in stream_history)))
            rows.append(
                {
                    "scenario": label,
                    "mode": "Streaming",
                    "tokens": stream_tokens,
                    "duration": stream_duration,
                    "throughput": stream_tokens / stream_duration if stream_duration > 0 else 0.0,
                }
            )

            logging.info("=== %s | Traditional generate (chat) ===", label)
            greedy_history, greedy_duration = run_traditional_chat(
                backend=backend, user_turns=user_turns, max_new_tokens=max_new_tokens
            )
            summarize_chat(
                mode=f"{label} | Traditional generate",
                history=greedy_history,
                duration=greedy_duration,
                backend=backend,
            )
            greedy_tokens = len(backend.tokenize(" ".join(r for _, r in greedy_history)))
            rows.append(
                {
                    "scenario": label,
                    "mode": "Traditional",
                    "tokens": greedy_tokens,
                    "duration": greedy_duration,
                    "throughput": greedy_tokens / greedy_duration if greedy_duration > 0 else 0.0,
                }
            )
        else:
            prompt = scenario["payload"]
            logging.info("=== %s | Streaming ===", label)
            stream_text, stream_duration = run_streaming(
                engine=engine, prompt=prompt, max_new_tokens=max_new_tokens
            )
            summarize(
                mode=f"{label} | Streaming",
                text=stream_text,
                duration=stream_duration,
                backend=backend,
            )
            stream_tokens = len(backend.tokenize(stream_text))
            rows.append(
                {
                    "scenario": label,
                    "mode": "Streaming",
                    "tokens": stream_tokens,
                    "duration": stream_duration,
                    "throughput": stream_tokens / stream_duration if stream_duration > 0 else 0.0,
                }
            )

            logging.info("=== %s | Traditional generate ===", label)
            greedy_text, greedy_duration = run_traditional(
                backend=backend, prompt=prompt, max_new_tokens=max_new_tokens
            )
            summarize(
                mode=f"{label} | Traditional generate",
                text=greedy_text,
                duration=greedy_duration,
                backend=backend,
            )
            greedy_tokens = len(backend.tokenize(greedy_text))
            rows.append(
                {
                    "scenario": label,
                    "mode": "Traditional",
                    "tokens": greedy_tokens,
                    "duration": greedy_duration,
                    "throughput": greedy_tokens / greedy_duration if greedy_duration > 0 else 0.0,
                }
            )

    print("\nBenchmark comparison (generated tokens only):")
    for scenario in scenarios:
        label = scenario["label"]
        stream_row = next(r for r in rows if r["scenario"] == label and r["mode"] == "Streaming")
        greedy_row = next(
            r for r in rows if r["scenario"] == label and r["mode"] == "Traditional"
        )
        speedup = (
            stream_row["throughput"] / greedy_row["throughput"]
            if greedy_row["throughput"] > 0
            else 0.0
        )
        delta = greedy_row["duration"] - stream_row["duration"]
        print(
            f"- {label}: streaming {stream_row['throughput']:.2f} tok/s vs"
            f" traditional {greedy_row['throughput']:.2f} tok/s"
            f" (x{speedup:.2f} throughput, Δ{delta:+.3f}s duration)"
        )
    print()


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
        or DEFAULT_MODEL
    )
    os.environ.setdefault("MODEL_NAME", model_name)

    backend = build_backend(model_name, ModelBackend, get_device)
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    token_budget = args.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
    user_turns = args.prompt

    if args.benchmark_suite or not user_turns:
        logging.info(
            "Running benchmark suite with model=%s max_new_tokens=%d",
            model_name,
            token_budget,
        )
        run_benchmark_suite(
            engine=engine, backend=backend, max_new_tokens=token_budget
        )
        print("Done. Use the logs above to trace each benchmark.")
        return

    logging.info(
        "Starting comparison with max_new_tokens=%d (prompts=%r)",
        token_budget,
        user_turns,
    )

    if args.multi_turn or len(user_turns) > 1:
        logging.info("=== Streaming multi-turn via mini-sglang (prefill + decode) ===")
        stream_history, stream_duration = run_streaming_chat(
            engine=engine, user_turns=user_turns, max_new_tokens=token_budget
        )
        summarize_chat(
            mode="Streaming", history=stream_history, duration=stream_duration, backend=backend
        )

        logging.info("=== Traditional multi-turn generate() ===")
        greedy_history, greedy_duration = run_traditional_chat(
            backend=backend, user_turns=user_turns, max_new_tokens=token_budget
        )
        summarize_chat(
            mode="Traditional generate", history=greedy_history, duration=greedy_duration, backend=backend
        )
    else:
        prompt = user_turns[0]
        logging.info("=== Streaming via mini-sglang (prefill + decode) ===")
        stream_text, stream_duration = run_streaming(
            engine=engine, prompt=prompt, max_new_tokens=token_budget
        )
        summarize(
            mode="Streaming", text=stream_text, duration=stream_duration, backend=backend
        )

        logging.info("=== Traditional single-call generate() ===")
        greedy_text, greedy_duration = run_traditional(
            backend=backend, prompt=prompt, max_new_tokens=token_budget
        )
        summarize(
            mode="Traditional generate", text=greedy_text, duration=greedy_duration, backend=backend
        )

    print("Done. Use the logs above to trace each step.")


if __name__ == "__main__":
    main()
