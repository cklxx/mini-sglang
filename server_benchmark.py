"""Benchmark the FastAPI server path (streaming) with TTFB and throughput.

This now runs three comparisons side-by-side so learners can see how the
mini-sglang HTTP server stacks up against an HF streaming baseline and an
in-process engine run (no HTTP overhead).
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Tuple

from fastapi.testclient import TestClient

from api.server import app, backend
from config import MAX_NEW_TOKENS_DEFAULT
from engine.engine import SGLangMiniEngine
from one_click_compare import ensure_dependencies


def run_server_stream(prompt: str, max_new_tokens: int, mode: str) -> Tuple[str, float, float]:
    client = TestClient(app)
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "stream": True,
        "mode": mode,
    }

    chunks: list[str] = []
    t_start = time.perf_counter()
    ttfb = None

    with client.stream("POST", "/generate", json=payload, timeout=None) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            msg = json.loads(line)
            if "text_delta" in msg:
                chunks.append(msg["text_delta"])
                if ttfb is None:
                    ttfb = time.perf_counter() - t_start
            elif msg.get("event") == "done":
                break

    ttfb = ttfb if ttfb is not None else time.perf_counter() - t_start
    duration = time.perf_counter() - t_start
    text = "".join(chunks)
    return text, ttfb, duration


def run_engine_stream(prompt: str, max_new_tokens: int) -> Tuple[str, float, float]:
    """Benchmark the in-process mini-sglang engine (no HTTP).

    This mirrors the "full" prefill + decode loop to show best-case throughput
    when request/response overhead is removed. It reuses the server backend so
    model weights are not reloaded.
    """

    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    chunks: list[str] = []
    t_start = time.perf_counter()
    ttfb: Optional[float] = None

    def on_stream(text_delta: str) -> None:
        nonlocal ttfb
        chunks.append(text_delta)
        if ttfb is None:
            ttfb = time.perf_counter() - t_start

    engine.run_generate(
        prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=on_stream
    )

    ttfb = ttfb if ttfb is not None else time.perf_counter() - t_start
    duration = time.perf_counter() - t_start
    text = "".join(chunks)
    return text, ttfb, duration


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the streaming HTTP server path.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain how streaming token generation reduces TTFB and improves UX for chat.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS_DEFAULT,
        help="Generation budget for the server benchmark",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip dependency auto-install (for pre-provisioned environments)",
    )
    args = parser.parse_args()

    if not args.no_bootstrap:
        ensure_dependencies()

    prompt_preview = args.prompt[:60]
    print(f"\nServer benchmark (streaming) for prompt={prompt_preview!r}:")

    text_sg, ttfb_sg, duration_sg = run_server_stream(
        args.prompt, args.max_new_tokens, mode="sglang"
    )
    tokens_sg = len(backend.tokenize(text_sg))
    throughput_sg = tokens_sg / duration_sg if duration_sg > 0 else 0.0

    text_hf, ttfb_hf, duration_hf = run_server_stream(
        args.prompt, args.max_new_tokens, mode="hf"
    )
    tokens_hf = len(backend.tokenize(text_hf))
    throughput_hf = tokens_hf / duration_hf if duration_hf > 0 else 0.0

    engine_text, engine_ttfb, engine_duration = run_engine_stream(
        prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )
    tokens_engine = len(backend.tokenize(engine_text))
    throughput_engine = tokens_engine / engine_duration if engine_duration > 0 else 0.0

    print(f"- HTTP server (sglang):      TTFB={ttfb_sg:.3f}s duration={duration_sg:.3f}s tokens={tokens_sg} throughput={throughput_sg:.2f} tok/s")
    print(f"- HTTP server (hf baseline): TTFB={ttfb_hf:.3f}s duration={duration_hf:.3f}s tokens={tokens_hf} throughput={throughput_hf:.2f} tok/s")
    print(f"- Engine (in-process mini):  TTFB={engine_ttfb:.3f}s duration={engine_duration:.3f}s tokens={tokens_engine} throughput={throughput_engine:.2f} tok/s")

    print("\nOutput previews:")
    print(f"- server (sglang): {text_sg[:120]!r}")
    print(f"- server (hf): {text_hf[:120]!r}")
    print(f"- engine (no HTTP): {engine_text[:120]!r}\n")


if __name__ == "__main__":
    main()
