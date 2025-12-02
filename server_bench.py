"""Benchmark streaming throughput across HTTP server paths."""
from __future__ import annotations

import argparse
import json
import time
from typing import Optional, Tuple

import httpx
import sglang as sgl
from fastapi.testclient import TestClient

from api.server import app as mini_app, backend as mini_backend
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME


def _count_tokens(text: str, tokenizer: Optional[object]) -> int:
    if tokenizer is None:
        return len(mini_backend.tokenize(text))
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))
    if callable(tokenizer):
        return len(tokenizer(text))
    return len(mini_backend.tokenize(text))


def stream_sglang_http(
    *, prompt: str, max_new_tokens: int, server_url: str, tokenizer: Optional[object]
) -> Tuple[str, float, float, int]:
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens},
        "stream": True,
    }

    chunks: list[str] = []
    start = time.perf_counter()
    ttfb: Optional[float] = None
    prev_len = 0

    with httpx.stream(
        "POST", f"{server_url.rstrip('/')}/generate", json=payload, timeout=None
    ) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            msg = json.loads(data)
            if "text" in msg:
                new_text: str = msg["text"]
                delta = new_text[prev_len:]
                if delta:
                    chunks.append(delta)
                    prev_len = len(new_text)
                    if ttfb is None:
                        ttfb = time.perf_counter() - start
            elif "error" in msg:
                raise RuntimeError(msg["error"])

    ttfb = ttfb if ttfb is not None else time.perf_counter() - start
    duration = time.perf_counter() - start
    text = "".join(chunks)
    tokens = _count_tokens(text, tokenizer)
    return text, ttfb, duration, tokens


def stream_mini_server(
    *, prompt: str, max_new_tokens: int, mode: str
) -> Tuple[str, float, float, int]:
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "stream": True,
        "mode": mode,
    }

    chunks: list[str] = []
    start = time.perf_counter()
    ttfb: Optional[float] = None
    client = TestClient(mini_app)

    with client.stream("POST", "/generate", json=payload, timeout=None) as resp:
        resp.raise_for_status()
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            msg = json.loads(line)
            if "text_delta" in msg:
                chunks.append(msg["text_delta"])
                if ttfb is None:
                    ttfb = time.perf_counter() - start
            elif msg.get("event") == "done":
                break

    ttfb = ttfb if ttfb is not None else time.perf_counter() - start
    duration = time.perf_counter() - start
    text = "".join(chunks)
    tokens = len(mini_backend.tokenize(text))
    return text, ttfb, duration, tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare streaming throughput across sglang server, mini-sglang FastAPI, and HF baseline"
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Explain how streaming token generation affects TTFB.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS_DEFAULT,
        help="Generation budget for all server paths",
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        help="Optional remote sglang server URL; defaults to launching a local Runtime",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name when starting a local sglang Runtime",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token_budget = args.max_new_tokens
    model_name = args.model or MODEL_NAME

    runtime: Optional[sgl.Runtime] = None
    runtime_tokenizer = None
    sglang_url = args.sglang_url.rstrip("/") if args.sglang_url else None
    if sglang_url is None:
        runtime = sgl.Runtime(
            model_path=model_name,
            tokenizer_path=model_name,
            trust_remote_code=True,
            log_level="error",
        )
        sglang_url = runtime.url
        runtime_tokenizer = runtime.get_tokenizer()

    try:
        sg_text, sg_ttfb, sg_duration, sg_tokens = stream_sglang_http(
            prompt=args.prompt,
            max_new_tokens=token_budget,
            server_url=sglang_url,
            tokenizer=runtime_tokenizer,
        )
    finally:
        if runtime is not None:
            runtime.shutdown()

    mini_text, mini_ttfb, mini_duration, mini_tokens = stream_mini_server(
        prompt=args.prompt, max_new_tokens=token_budget, mode="sglang"
    )
    hf_text, hf_ttfb, hf_duration, hf_tokens = stream_mini_server(
        prompt=args.prompt, max_new_tokens=token_budget, mode="hf"
    )

    print(
        f"Server benchmark prompt={args.prompt!r} | max_new_tokens={token_budget} | model={model_name}"
    )
    print(
        f"- sglang server:   TTFB={sg_ttfb:.3f}s duration={sg_duration:.3f}s tokens={sg_tokens} "
        f"throughput={sg_tokens/sg_duration if sg_duration>0 else 0:.2f} tok/s"
    )
    print(
        f"- mini-sglang API: TTFB={mini_ttfb:.3f}s duration={mini_duration:.3f}s tokens={mini_tokens} "
        f"throughput={mini_tokens/mini_duration if mini_duration>0 else 0:.2f} tok/s"
    )
    print(
        f"- HF baseline API: TTFB={hf_ttfb:.3f}s duration={hf_duration:.3f}s tokens={hf_tokens} "
        f"throughput={hf_tokens/hf_duration if hf_duration>0 else 0:.2f} tok/s"
    )

    print("\nPreviews:")
    print(f"- sglang:   {sg_text[:120]!r}")
    print(f"- mini:     {mini_text[:120]!r}")
    print(f"- hf:       {hf_text[:120]!r}")


if __name__ == "__main__":
    main()
