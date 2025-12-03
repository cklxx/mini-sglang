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

# Default to a longer local benchmark (more tokens + repeated runs) so GPU
# paths have enough work to shine by default.
LOCAL_BENCH_MAX_NEW_TOKENS = max(1024, MAX_NEW_TOKENS_DEFAULT)
LOCAL_BENCH_REPEAT = 3

@sgl.function
def _sglang_single_turn(s, prompt: str):
    s += sgl.user(cast(Any, prompt))
    s += sgl.assistant(sgl.gen("answer"))


def _init_sglang_runtime(model_name: str):
    runtime = sgl.Runtime(
        model_path=model_name,
        tokenizer_path=model_name,
        trust_remote_code=True,
        log_level="error",
    )
    # Some sglang builds leave pid unset; guard __del__/shutdown from exploding.
    runtime.pid = getattr(runtime, "pid", None)
    return runtime


def run_sglang_runtime(
    *,
    prompt: str,
    max_new_tokens: int,
    model_name: str,
    runtime: Any | None = None,
) -> Tuple[str, float, int]:
    """Stream tokens through a real sglang Runtime launched in-process."""

    owns_runtime = runtime is None
    runtime = runtime or _init_sglang_runtime(model_name)
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
    if owns_runtime:
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
    required = ("uvloop", "psutil", "triton", "PIL", "partial_json_parser")
    return [mod for mod in required if importlib.util.find_spec(mod) is None]


def _summarize_runs(
    runs: list[Tuple[str, float, int]]
) -> Tuple[str, float, float, float, int]:
    if not runs:
        return "", 0.0, 0.0, 0.0, 0
    total_tokens = sum(tokens for _, _, tokens in runs)
    total_duration = sum(duration for _, duration, _ in runs)
    avg_duration = total_duration / len(runs)
    throughput = total_tokens / total_duration if total_duration > 0 else 0.0
    return runs[0][0], total_duration, avg_duration, throughput, total_tokens


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
        help=(
            "Override generation budget "
            f"(defaults to a longer local bench: {LOCAL_BENCH_MAX_NEW_TOKENS})"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for all three paths (defaults to MODEL_NAME)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=LOCAL_BENCH_REPEAT,
        help="Number of timed runs per backend for a longer, steadier benchmark",
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
    token_budget = args.max_new_tokens or LOCAL_BENCH_MAX_NEW_TOKENS
    repeat = max(1, args.repeat)

    backend = ModelBackend(model_name=model_name, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    print(
        f"Benchmarking prompt={args.prompt!r} | model={model_name} | "
        f"max_new_tokens={token_budget} | repeat={repeat}"
    )

    skip_sg = os.getenv("SKIP_SGLANG_RUNTIME", "0") == "1"
    sg_error: str | None = "Skipped via SKIP_SGLANG_RUNTIME" if skip_sg else None
    sg_runs: list[Tuple[str, float, int]] = []
    sg_text = ""
    runtime = None
    if not skip_sg:
        missing = _missing_sglang_runtime_deps()
        if missing:
            sg_error = (
                "Missing sglang runtime deps: " + ", ".join(sorted(missing)) + "; "
                "install the extras or set SKIP_SGLANG_RUNTIME=1"
            )
        else:
            try:
                runtime = _init_sglang_runtime(model_name)
                try:
                    sg_runs = [
                        run_sglang_runtime(
                            prompt=args.prompt,
                            max_new_tokens=token_budget,
                            model_name=model_name,
                            runtime=runtime,
                        )
                        for _ in range(repeat)
                    ]
                finally:
                    runtime.shutdown()
            except Exception as exc:  # pragma: no cover - best-effort guard
                sg_error = f"sglang Runtime failed: {exc}"

    mini_runs = [
        run_mini_sglang(
            engine=engine,
            prompt=args.prompt,
            max_new_tokens=token_budget,
            backend=backend,
        )
        for _ in range(repeat)
    ]
    hf_runs = [
        run_hf_streaming(
            backend=backend, prompt=args.prompt, max_new_tokens=token_budget
        )
        for _ in range(repeat)
    ]

    sg_text, sg_duration, sg_avg_duration, sg_throughput, sg_tokens = _summarize_runs(
        sg_runs
    )
    mini_text, mini_duration, mini_avg_duration, mini_throughput, mini_tokens = (
        _summarize_runs(mini_runs)
    )
    hf_text, hf_duration, hf_avg_duration, hf_throughput, hf_tokens = _summarize_runs(
        hf_runs
    )

    print("\nLocal results:")
    if sg_runs:
        print(
            f"- sglang Runtime: runs={len(sg_runs)} tokens={sg_tokens} "
            f"duration={sg_duration:.3f}s avg/run={sg_avg_duration:.3f}s "
            f"throughput={sg_throughput:.2f} tok/s"
        )
    else:
        print(f"- sglang Runtime: skipped ({sg_error})")
    print(
        f"- mini-sglang:    runs={len(mini_runs)} tokens={mini_tokens} "
        f"duration={mini_duration:.3f}s avg/run={mini_avg_duration:.3f}s "
        f"throughput={mini_throughput:.2f} tok/s"
    )
    print(
        f"- HF streaming:   runs={len(hf_runs)} tokens={hf_tokens} "
        f"duration={hf_duration:.3f}s avg/run={hf_avg_duration:.3f}s "
        f"throughput={hf_throughput:.2f} tok/s"
    )

    print("\nPreviews:")
    if sg_runs:
        print(f"- sglang:   {sg_text[:120]!r}")
    else:
        print(f"- sglang:   [skipped]")
    print(f"- mini:     {mini_text[:120]!r}")
    print(f"- hf:       {hf_text[:120]!r}")


if __name__ == "__main__":
    main()
