"""Benchmark the FastAPI server path (streaming) with TTFB and throughput."""

from __future__ import annotations

import argparse
import json
import time
from typing import Tuple

from fastapi.testclient import TestClient

from api.server import app, backend
from config import MAX_NEW_TOKENS_DEFAULT


def run_server_stream(prompt: str, max_new_tokens: int) -> Tuple[str, float, float]:
    client = TestClient(app)
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True}

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
    args = parser.parse_args()

    text, ttfb, duration = run_server_stream(args.prompt, args.max_new_tokens)
    tokens = len(backend.tokenize(text))
    throughput = tokens / duration if duration > 0 else 0.0

    print("\nServer benchmark (streaming):")
    print(f"- Prompt preview: {args.prompt[:60]!r}")
    print(f"- TTFB: {ttfb:.3f}s | duration: {duration:.3f}s | tokens: {tokens} | throughput: {throughput:.2f} tok/s")
    print(f"- Output preview: {text[:120]!r}\n")


if __name__ == "__main__":
    main()
