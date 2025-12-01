"""Benchmark streaming mini-sglang vs vanilla transformer generate.

This mirrors the profiling loops you might run on full sglang to contrast
step-wise streaming against batched generation.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.model_backend import ModelBackend
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark streaming mini-sglang vs vanilla generate"
    )
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default="Benchmarking mini-sglang",
        help="Prompt to feed the model",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        dest="max_new_tokens",
        help="Override the default token budget",
    )
    return parser.parse_args()


def timed_stream(engine: SGLangMiniEngine, prompt: str, max_new_tokens: Optional[int]):
    chunks: list[str] = []
    start = time.perf_counter()

    def stream_callback(text_delta: str) -> None:
        chunks.append(text_delta)

    engine.run_generate(prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=stream_callback)
    duration = time.perf_counter() - start
    text = "".join(chunks)
    return text, duration


def timed_vanilla_generate(
    backend: ModelBackend, prompt: str, max_new_tokens: int
):
    prompt_ids = backend.tokenize(prompt)
    start = time.perf_counter()
    generated_ids = backend.generate_greedy(
        prompt_ids=prompt_ids, max_new_tokens=max_new_tokens
    )
    text = backend.decode_tokens(generated_ids)
    duration = time.perf_counter() - start
    return text, duration


def main(prompt: str, max_new_tokens: Optional[int]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    print(f"Benchmarking model={MODEL_NAME} on device={backend.device}")
    print(f"Prompt: {prompt!r}\n")

    token_budget = max_new_tokens or MAX_NEW_TOKENS_DEFAULT

    stream_text, stream_duration = timed_stream(
        engine=engine, prompt=prompt, max_new_tokens=token_budget
    )
    stream_tokens = len(backend.tokenize(stream_text))

    vanilla_text, vanilla_duration = timed_vanilla_generate(
        backend=backend, prompt=prompt, max_new_tokens=token_budget
    )
    vanilla_tokens = len(backend.tokenize(vanilla_text))

    print("Streaming mode:")
    print(f"  tokens: {stream_tokens}")
    print(f"  duration: {stream_duration:.3f}s")
    if stream_duration > 0:
        print(f"  throughput: {stream_tokens/stream_duration:.2f} tok/s")

    print("\nVanilla transformer generate (non-sglang baseline):")
    print(f"  tokens: {vanilla_tokens}")
    print(f"  duration: {vanilla_duration:.3f}s")
    if vanilla_duration > 0:
        print(f"  throughput: {vanilla_tokens/vanilla_duration:.2f} tok/s")

    print("\nOutput preview (streaming):", repr(stream_text[:120]))
    print("Output preview (vanilla):", repr(vanilla_text[:120]))


if __name__ == "__main__":
    args = parse_args()
    main(prompt=args.prompt, max_new_tokens=args.max_new_tokens)
