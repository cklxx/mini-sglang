"""Concurrent benchmark suite (single entrypoint, no CLI flags).

Runs mixed workloads concurrently to exercise scheduling, cache reuse, and
throughput under load. Defaults are sensible; tune via env vars only.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Optional

from backend.model_backend import ModelBackend
from backend.hf_runner import HFBaseline
from config import MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine


@dataclass
class Workload:
    name: str
    prompt_tokens: int
    max_new_tokens: int


def _make_prompt(token_count: int) -> str:
    # Tokenizer dependent, but a repeated short word keeps tokenization stable.
    return ("hello " * (token_count // 2 + 1)).strip()


def _default_workloads() -> list[Workload]:
    return [
        Workload("long", prompt_tokens=256, max_new_tokens=1024),
    ]


def _run_once(
    name: str,
    prompt: str,
    max_new_tokens: int,
    engine: SGLangMiniEngine,
    backend: ModelBackend,
    hf_runner: HFBaseline,
    use_hf: bool,
) -> tuple[float, float, int]:
    start = time.perf_counter()
    first: Optional[float] = None
    tokens: Optional[int] = None

    def cb(delta: str) -> None:
        nonlocal first
        if delta and first is None:
            first = time.perf_counter()

    if use_hf:
        text, _ = hf_runner.generate_streaming(
            prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=cb
        )
        tokens = len(hf_runner.tokenize(text))
    else:
        text = engine.run_generate(
            prompt=prompt, max_new_tokens=max_new_tokens, stream_callback=cb
        )
        tokens = len(backend.tokenize(text))

    end = time.perf_counter()
    ttfb = (first - start) if first is not None else (end - start)
    duration = end - start
    return ttfb, duration, tokens


def _summarize(samples: list[tuple[float, float, int]]) -> tuple[float, float, float, int]:
    if not samples:
        return 0.0, 0.0, 0.0, 0
    ttfbs = [s[0] for s in samples]
    durs = [s[1] for s in samples]
    toks = [s[2] for s in samples]
    total_tokens = sum(toks)
    total_time = sum(durs)
    throughput = total_tokens / total_time if total_time > 0 else 0.0
    p50_ttfb = statistics.median(ttfbs)
    p95_ttfb = statistics.quantiles(ttfbs, n=100)[94] if len(ttfbs) >= 2 else p50_ttfb
    return p50_ttfb, p95_ttfb, throughput, total_tokens


def _run_suite(
    *,
    backend: ModelBackend,
    engine: SGLangMiniEngine,
    hf_runner: HFBaseline,
    workloads: Iterable[Workload],
    repeat: int,
    warmup: int,
    concurrency: int,
    mixed_prompts: bool,
) -> None:
    # Warmup to hit graph/static cache etc.
    if warmup > 0:
        wp = Workload("warmup", prompt_tokens=8, max_new_tokens=32)
        prompt = _make_prompt(wp.prompt_tokens)
        for _ in range(warmup):
            engine.run_generate(
                prompt=prompt,
                max_new_tokens=wp.max_new_tokens,
                stream_callback=lambda _: None,
            )

    print("Concurrent benchmark:")
    for wl in workloads:
        prompts = []
        for i in range(concurrency):
            if mixed_prompts and i % 2 == 1:
                prompts.append(f"{_make_prompt(wl.prompt_tokens)} #{random.randint(0, 9999)}")
            else:
                prompts.append(_make_prompt(wl.prompt_tokens))

        mini_samples: list[tuple[float, float, int]] = []
        for i in range(concurrency * repeat):
            print(f"  [mini] {wl.name} run {i + 1}/{concurrency * repeat}")
            mini_samples.append(
                _run_once(
                    f"{wl.name}-{i}",
                    prompts[i % len(prompts)],
                    wl.max_new_tokens,
                    engine,
                    backend,
                    hf_runner,
                    False,
                )
            )

        hf_samples: list[tuple[float, float, int]] = []
        for i in range(repeat):
            print(f"  [hf] {wl.name} run {i + 1}/{repeat}")
            hf_samples.append(
                _run_once(
                    f"{wl.name}-hf-{i}",
                    prompts[i % len(prompts)],
                    wl.max_new_tokens,
                    engine,
                    backend,
                    hf_runner,
                    True,
                )
            )

        mini_p50_ttfb, mini_p95_ttfb, mini_tp, mini_tokens = _summarize(mini_samples)
        hf_p50_ttfb, hf_p95_ttfb, hf_tp, hf_tokens = _summarize(hf_samples)

        print(
            f"- {wl.name}: prompt_tokens={wl.prompt_tokens} max_new_tokens={wl.max_new_tokens} "
            f"repeat={repeat} concurrency={concurrency} mixed_prompts={mixed_prompts}"
        )
        print(
            f"  mini-sglang: tokens={mini_tokens} p50_ttfb={mini_p50_ttfb:.3f}s "
            f"p95_ttfb={mini_p95_ttfb:.3f}s throughput={mini_tp:.2f} tok/s"
        )
        print(
            f"  HF stream:   tokens={hf_tokens} p50_ttfb={hf_p50_ttfb:.3f}s "
            f"p95_ttfb={hf_p95_ttfb:.3f}s throughput={hf_tp:.2f} tok/s"
        )


def main() -> None:
    repeat = max(1, int(os.getenv("BENCH_REPEAT", "1")))
    warmup = max(0, int(os.getenv("BENCH_WARMUP", "0")))
    concurrency = max(1, int(os.getenv("BENCH_CONCURRENCY", "1")))
    mixed_prompts = os.getenv("BENCH_MIXED_PROMPTS", "1") != "0"
    log_level = os.getenv("BENCH_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
    hf_runner = HFBaseline(model_name=MODEL_NAME, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=max(w.max_new_tokens for w in _default_workloads())
    )

    _run_suite(
        backend=backend,
        engine=engine,
        hf_runner=hf_runner,
        workloads=_default_workloads(),
        repeat=repeat,
        warmup=warmup,
        concurrency=concurrency,
        mixed_prompts=mixed_prompts,
    )


if __name__ == "__main__":
    main()
