"""One-click benchmark: sglang streaming vs HF streaming vs HTTP server.

This mirrors sglang's bench mode in a minimal way:
- Uses the in-process engine for sglang streaming (prefill + decode).
- Uses HF TextIteratorStreamer as the non-sglang streaming baseline.
- Hits the FastAPI server via TestClient to measure TTFB + throughput.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any, Tuple

from fastapi.testclient import TestClient

from api.server import app, backend as server_backend
from config import MAX_NEW_TOKENS_DEFAULT
from one_click_compare import (
    build_backend,
    ensure_dependencies,
    load_components,
    run_baseline_streaming,
    run_streaming,
)


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


def summarize(label: str, text: str, duration: float, backend: Any) -> dict[str, Any]:
    tokens = len(backend.tokenize(text))
    tp = tokens / duration if duration > 0 else 0.0
    return {"label": label, "tokens": tokens, "duration": duration, "throughput": tp}


def main() -> None:
    parser = argparse.ArgumentParser(description="One-click benchmark (sglang vs HF baseline vs server)")
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

    ModelBackend, SGLangMiniEngine, _, MODEL_NAME, get_device = load_components()
    backend = build_backend(
        MODEL_NAME, ModelBackend, get_device, compile_model=False
    )
    engine = SGLangMiniEngine(backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT)

    print(f"Running benchmark with model={MODEL_NAME} max_new_tokens={args.max_new_tokens}")
    print(f"Prompt: {args.prompt!r}\n")

    s_text, s_duration = run_streaming(
        engine=engine, prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )
    s_metrics = summarize("sglang streaming", s_text, s_duration, backend)

    b_text, b_duration = run_baseline_streaming(
        backend=backend, prompt=args.prompt, max_new_tokens=args.max_new_tokens
    )
    b_metrics = summarize("HF streaming baseline", b_text, b_duration, backend)

    sv_text, sv_ttfb, sv_tp = run_server_stream(args.prompt, args.max_new_tokens, mode="sglang")
    sv_hf_text, sv_hf_ttfb, sv_hf_tp = run_server_stream(args.prompt, args.max_new_tokens, mode="hf")

    print("Results:")
    print(f"- sglang streaming:  throughput={s_metrics['throughput']:.2f} tok/s  duration={s_metrics['duration']:.3f}s")
    print(f"- HF streaming baseline: throughput={b_metrics['throughput']:.2f} tok/s  duration={b_metrics['duration']:.3f}s")
    print(f"- HTTP server (sglang): throughput={sv_tp:.2f} tok/s  TTFB={sv_ttfb:.3f}s")
    print(f"- HTTP server (hf baseline): throughput={sv_hf_tp:.2f} tok/s  TTFB={sv_hf_ttfb:.3f}s")
    print("\nPreviews:")
    print(f"- sglang: {s_text[:120]!r}")
    print(f"- HF baseline: {b_text[:120]!r}")
    print(f"- server (sglang): {sv_text[:120]!r}")
    print(f"- server (hf): {sv_hf_text[:120]!r}")


if __name__ == "__main__":
    main()
