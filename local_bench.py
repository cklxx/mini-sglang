"""Benchmark local inference across sglang, mini-sglang, and HF baselines."""
from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import time
from typing import Any, Tuple, cast

import sglang as sgl

from backend.model_backend import ModelBackend
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine


@sgl.function
def _sglang_single_turn(s, prompt: str):
    s += sgl.user(cast(Any, prompt))
    s += sgl.assistant(sgl.gen("answer"))


def run_sglang_runtime(
    *, prompt: str, max_new_tokens: int, model_name: str
) -> Tuple[str, float, int]:
    """Stream tokens through a real sglang Runtime launched in-process."""

    runtime = sgl.Runtime(
        model_path=model_name,
        tokenizer_path=model_name,
        trust_remote_code=True,
        log_level="error",
    )

    tokenizer = runtime.get_tokenizer()
    start = time.perf_counter()
    sgl_fn = cast(Any, _sglang_single_turn)
    state = sgl_fn.run(
        prompt,
        backend=runtime,
        max_new_tokens=max_new_tokens,
        stream=True,
    )
    chunks: list[str] = []
    for delta in state.text_iter("answer"):
        if delta:
            chunks.append(delta)
    duration = time.perf_counter() - start
    text = "".join(chunks)
    tokens = len(tokenizer.encode(text)) if duration > 0 else 0
    runtime.shutdown()
    return text, duration, tokens


def run_mini_sglang(
    *, engine: SGLangMiniEngine, prompt: str, max_new_tokens: int, backend: ModelBackend
) -> Tuple[str, float, int]:
    chunks: list[str] = []
    start = time.perf_counter()

    def on_stream(text_delta: str) -> None:
        chunks.append(text_delta)

    engine.run_generate(
        prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=on_stream
    )
    duration = time.perf_counter() - start
    text = "".join(chunks)
    tokens = len(backend.tokenize(text)) if duration > 0 else 0
    return text, duration, tokens


def run_hf_streaming(
    *, backend: ModelBackend, prompt: str, max_new_tokens: int
) -> Tuple[str, float, int]:
    prompt_ids = backend.tokenize(prompt)
    start = time.perf_counter()
    text, _ = backend.generate_streaming_baseline(
        prompt_ids=prompt_ids, max_new_tokens=max_new_tokens
    )
    duration = time.perf_counter() - start
    tokens = len(backend.tokenize(text)) if duration > 0 else 0
    return text, duration, tokens


def _missing_sglang_runtime_deps() -> list[str]:
    required = ("uvloop", "psutil", "triton")
    return [mod for mod in required if importlib.util.find_spec(mod) is None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Local benchmark: sglang Runtime vs mini-sglang vs HF streaming baseline"
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain how streaming reduces latency for chat users.",
        help="Prompt to benchmark",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        dest="max_new_tokens",
        help="Override generation budget (defaults to project setting)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for all three paths (defaults to MODEL_NAME)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    model_name = args.model or MODEL_NAME
    token_budget = args.max_new_tokens or MAX_NEW_TOKENS_DEFAULT

    backend = ModelBackend(model_name=model_name, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    print(
        f"Benchmarking prompt={args.prompt!r} | model={model_name} | max_new_tokens={token_budget}"
    )

    skip_sg = os.getenv("SKIP_SGLANG_RUNTIME", "0") == "1"
    sg_error: str | None = "Skipped via SKIP_SGLANG_RUNTIME" if skip_sg else None
    sg_result: Tuple[str, float, int] | None = None
    sg_text = ""
    if not skip_sg:
        missing = _missing_sglang_runtime_deps()
        if missing:
            sg_error = (
                "Missing sglang runtime deps: " + ", ".join(sorted(missing)) + "; "
                "install the extras or set SKIP_SGLANG_RUNTIME=1"
            )
        else:
            try:
                sg_result = run_sglang_runtime(
                    prompt=args.prompt, max_new_tokens=token_budget, model_name=model_name
                )
            except Exception as exc:  # pragma: no cover - best-effort guard
                sg_error = f"sglang Runtime failed: {exc}"

    mini_text, mini_duration, mini_tokens = run_mini_sglang(
        engine=engine,
        prompt=args.prompt,
        max_new_tokens=token_budget,
        backend=backend,
    )
    hf_text, hf_duration, hf_tokens = run_hf_streaming(
        backend=backend, prompt=args.prompt, max_new_tokens=token_budget
    )

    print("\nLocal results:")
    if sg_result is not None:
        sg_text, sg_duration, sg_tokens = sg_result
        print(
            f"- sglang Runtime: tokens={sg_tokens} duration={sg_duration:.3f}s "
            f"throughput={sg_tokens/sg_duration if sg_duration>0 else 0:.2f} tok/s"
        )
    else:
        print(f"- sglang Runtime: skipped ({sg_error})")
    print(
        f"- mini-sglang:    tokens={mini_tokens} duration={mini_duration:.3f}s "
        f"throughput={mini_tokens/mini_duration if mini_duration>0 else 0:.2f} tok/s"
    )
    print(
        f"- HF streaming:   tokens={hf_tokens} duration={hf_duration:.3f}s "
        f"throughput={hf_tokens/hf_duration if hf_duration>0 else 0:.2f} tok/s"
    )

    print("\nPreviews:")
    if sg_result is not None:
        print(f"- sglang:   {sg_text[:120]!r}")
    else:
        print(f"- sglang:   [skipped]")
    print(f"- mini:     {mini_text[:120]!r}")
    print(f"- hf:       {hf_text[:120]!r}")


if __name__ == "__main__":
    main()
