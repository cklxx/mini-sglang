"""One-click benchmark: compare three server paths side by side.

This mirrors sglang's bench_serving.py pattern in a minimal way:
- Hits three streaming HTTP servers (sglang, HF baseline, mini-sglang) to log
  TTFB and throughput.
- Defaults to the in-process FastAPI app if no remote URLs are supplied, so you
  can still see three server-style measurements locally.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Tuple

import httpx

from fastapi.testclient import TestClient

from api.server import app, backend as server_backend
from config import MAX_NEW_TOKENS_DEFAULT
from one_click_compare import ensure_dependencies


def run_server_stream(
    prompt: str,
    max_new_tokens: int,
    mode: str,
    server_url: Optional[str] = None,
) -> Tuple[str, float, float]:
    """Stream from either the in-process FastAPI app or a remote server.

    This mirrors sglang's bench_serving.py pattern where three different
    servers (sglang, HF baseline, and a mini-sglang deployment) are compared
    head-to-head. If a server URL is provided we use HTTPX streaming; otherwise
    we fall back to the local TestClient.
    """

    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "stream": True, "mode": mode}

    chunks: list[str] = []
    t_start = time.perf_counter()
    ttfb = None

    def handle_lines(lines) -> None:
        nonlocal ttfb
        for raw_line in lines:
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

    if server_url:
        with httpx.stream("POST", f"{server_url.rstrip('/')}/generate", json=payload, timeout=None) as resp:
            resp.raise_for_status()
            handle_lines(resp.iter_lines())
    else:
        client = TestClient(app)
        with client.stream("POST", "/generate", json=payload, timeout=None) as resp:
            resp.raise_for_status()
            handle_lines(resp.iter_lines())

    ttfb = ttfb if ttfb is not None else time.perf_counter() - t_start
    duration = time.perf_counter() - t_start
    text = "".join(chunks)
    tokens = len(server_backend.tokenize(text))
    throughput = tokens / duration if duration > 0 else 0.0
    return text, ttfb, throughput
def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-click server benchmark (sglang vs HF baseline vs mini-sglang), "
            "mirroring sglang's bench_serving.py layout."
        )
    )
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
        "--sglang-url",
        help="Optional remote sglang server URL (defaults to local FastAPI app)",
    )
    parser.add_argument(
        "--hf-url",
        help="Optional remote HF baseline server URL (defaults to local FastAPI app)",
    )
    parser.add_argument(
        "--mini-url",
        help=(
            "Optional remote mini-sglang server URL. If omitted, this reuses the "
            "local FastAPI app so all three server paths are exercised."
        ),
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
        args.prompt, args.max_new_tokens, mode="sglang", server_url=args.sglang_url
    )
    sv_hf_text, sv_hf_ttfb, sv_hf_tp = run_server_stream(
        args.prompt, args.max_new_tokens, mode="hf", server_url=args.hf_url
    )
    mini_text, mini_ttfb, mini_tp = run_server_stream(
        args.prompt,
        args.max_new_tokens,
        mode="sglang",
        server_url=args.mini_url if args.mini_url is not None else args.sglang_url,
    )

    print("Server results (FastAPI /generate):")
    print(
        f"- HTTP server (sglang):      throughput={sv_tp:.2f} tok/s  TTFB={sv_ttfb:.3f}s"
    )
    print(
        f"- HTTP server (hf baseline): throughput={sv_hf_tp:.2f} tok/s  TTFB={sv_hf_ttfb:.3f}s"
    )
    print(
        f"- HTTP server (mini-sglang): throughput={mini_tp:.2f} tok/s  TTFB={mini_ttfb:.3f}s"
    )
    print("\nPreviews:")
    print(f"- server (sglang): {sv_text[:120]!r}")
    print(f"- server (hf): {sv_hf_text[:120]!r}")
    print(f"- server (mini): {mini_text[:120]!r}")


if __name__ == "__main__":
    main()
