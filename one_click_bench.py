"""One-click benchmark: sglang streaming vs HF streaming vs HTTP server.

This mirrors sglang's bench mode in a minimal way for the HTTP server:
- Hits the FastAPI server via TestClient to measure TTFB + throughput.
- Compares sglang engine mode vs HF streaming baseline mode on the server.
- Also times an in-process mini-sglang engine run (no HTTP) so learners can
  see all three paths side by side.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Tuple

from fastapi.testclient import TestClient

from api.server import app, backend as server_backend
from engine.engine import SGLangMiniEngine
from config import MAX_NEW_TOKENS_DEFAULT
from one_click_compare import ensure_dependencies


def run_server_stream(prompt: str, max_new_tokens: int, mode: str) -> Tuple[str, float, float]:
    client = TestClient(app)
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True, "mode": mode}

    chunks: list[str] = []
    t_start = time.perf_counter()
    ttfb = None

    with client.stream("POST", "/generate", json=payload, timeout=None) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line if isinstance(raw_line, str) else raw_line.decode("utf-8")
            line = line.strip()
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
    tokens = len(server_backend.tokenize(text))
    throughput = tokens / duration if duration > 0 else 0.0
    return text, ttfb, throughput


def run_engine_stream(prompt: str, max_new_tokens: int) -> Tuple[str, float, float]:
    """Benchmark the in-process mini-sglang engine without HTTP overhead."""

    engine = SGLangMiniEngine(
        backend=server_backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT
    )

    chunks: list[str] = []
    t_start = time.perf_counter()
    ttfb = None

    def stream_callback(text_delta: str) -> None:
        nonlocal ttfb
        chunks.append(text_delta)
        if ttfb is None:
            ttfb = time.perf_counter() - t_start

    engine.run_generate(
        prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=stream_callback
    )

    ttfb = ttfb if ttfb is not None else time.perf_counter() - t_start
    duration = time.perf_counter() - t_start
    text = "".join(chunks)
    tokens = len(server_backend.tokenize(text))
    throughput = tokens / duration if duration > 0 else 0.0
    return text, ttfb, throughput


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click server benchmark (sglang vs HF baseline)")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain how streaming token generation reduces TTFB for chat users.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS_DEFAULT,
        help="Token budget for all modes",
    )
    parser.add_argument(
        "--no-bootstrap",
        action="store_true",
        help="Skip dependency auto-install (for pre-provisioned environments)",
    )
    args = parser.parse_args()

    if not args.no_bootstrap:
        ensure_dependencies()

    print(f"Running server benchmark with max_new_tokens={args.max_new_tokens}")
    print(f"Prompt: {args.prompt!r}\n")

    sv_text, sv_ttfb, sv_tp = run_server_stream(
        args.prompt, args.max_new_tokens, mode="sglang"
    )
    sv_hf_text, sv_hf_ttfb, sv_hf_tp = run_server_stream(
        args.prompt, args.max_new_tokens, mode="hf"
    )
    engine_text, engine_ttfb, engine_tp = run_engine_stream(
        args.prompt, args.max_new_tokens
    )

    print("Server results (FastAPI /generate):")
    print(
        f"- HTTP server (sglang):      throughput={sv_tp:.2f} tok/s  TTFB={sv_ttfb:.3f}s"
    )
    print(
        f"- HTTP server (hf baseline): throughput={sv_hf_tp:.2f} tok/s  TTFB={sv_hf_ttfb:.3f}s"
    )
    print(
        f"- Engine (mini-sglang):      throughput={engine_tp:.2f} tok/s  TTFB={engine_ttfb:.3f}s"
    )
    print("\nPreviews:")
    print(f"- server (sglang): {sv_text[:120]!r}")
    print(f"- server (hf): {sv_hf_text[:120]!r}")
    print(f"- engine (mini): {engine_text[:120]!r}")


if __name__ == "__main__":
    main()
