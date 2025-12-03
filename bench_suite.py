"""Concurrent benchmark suite (single entrypoint, no CLI flags).

Runs mixed workloads concurrently to exercise scheduling, cache reuse, and
throughput under load. Defaults are sensible; tune via env vars only.

Optional: compare real sglang Runtime (sgl-kernel) vs HF streaming baseline for a
single prompt/length; enabled via BENCH_INCLUDE_SGLANG=1 and uses fixed defaults.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from backend.model_backend import ModelBackend
from config import MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine

try:
    from sglang import Runtime, function, gen, set_default_backend
except Exception as exc:  # pragma: no cover - optional at install time
    Runtime = None  # type: ignore
    function = None  # type: ignore
    gen = None  # type: ignore
    set_default_backend = None  # type: ignore
    _sglang_import_error = exc
else:
    _sglang_import_error = None


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
        Workload("short", prompt_tokens=16, max_new_tokens=256),
        Workload("medium", prompt_tokens=64, max_new_tokens=512),
        Workload("long", prompt_tokens=256, max_new_tokens=1024),
    ]


def _run_once(
    name: str,
    prompt: str,
    max_new_tokens: int,
    engine: SGLangMiniEngine,
    backend: ModelBackend,
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
        prompt_ids = backend.tokenize(prompt)
        text, _ = backend.generate_streaming_baseline(
            prompt_ids=prompt_ids, max_new_tokens=max_new_tokens, stream_callback=cb
        )
        tokens = len(backend.tokenize(text))
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
    p50_dur = statistics.median(durs)
    p95_dur = statistics.quantiles(durs, n=100)[94] if len(durs) >= 2 else p50_dur
    return p50_ttfb, p95_ttfb, throughput, total_tokens


def _run_suite(
    *,
    backend: ModelBackend,
    engine: SGLangMiniEngine,
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
            engine.run_generate(prompt=prompt, max_new_tokens=wp.max_new_tokens, stream_callback=lambda _: None)

    print("Concurrent benchmark:")
    for wl in workloads:
        prompts = []
        for i in range(concurrency):
            if mixed_prompts and i % 2 == 1:
                prompts.append(f"{_make_prompt(wl.prompt_tokens)} #{random.randint(0, 9999)}")
            else:
                prompts.append(_make_prompt(wl.prompt_tokens))

        def run_batch(use_hf: bool) -> list[tuple[float, float, int]]:
            samples: list[tuple[float, float, int]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = [
                    pool.submit(
                        _run_once,
                        f"{wl.name}-{i}",
                        prompts[i % len(prompts)],
                        wl.max_new_tokens,
                        engine,
                        backend,
                        use_hf,
                    )
                    for i in range(concurrency * repeat)
                ]
                for fut in concurrent.futures.as_completed(futs):
                    samples.append(fut.result())
            return samples

        mini_samples = run_batch(False)
        hf_samples = run_batch(True)

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


def _auto_dtype() -> Optional[torch.dtype]:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if torch.backends.mps.is_available():
        return torch.float16
    return None


def _p50(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0


def _bench_hf_stream(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    repeat: int,
    dtype: Optional[torch.dtype],
) -> Tuple[float, float]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    samples: List[Tuple[float, float, int]] = []
    for _ in range(repeat):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        t0 = time.perf_counter()
        first: Optional[float] = None
        thread = threading.Thread(
            target=model.generate,
            kwargs=dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                do_sample=False,
            ),
        )
        thread.start()
        chunks: List[str] = []
        for delta in streamer:
            if delta and first is None:
                first = time.perf_counter()
            chunks.append(delta)
        thread.join()
        t1 = time.perf_counter()
        tokens = len(tokenizer.encode("".join(chunks), add_special_tokens=False))
        samples.append(((first or t1) - t0, t1 - t0, tokens))
    ttfb_p50 = _p50([s[0] for s in samples])
    tp_p50 = _p50([s[2] / s[1] for s in samples if s[1] > 0])
    return ttfb_p50, tp_p50


def _bench_sglang_runtime(
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    repeat: int,
    tensor_parallel_size: int,
) -> Tuple[float, float]:
    if Runtime is None or function is None or gen is None or set_default_backend is None:
        raise RuntimeError(f"sglang not available: {_sglang_import_error}")
    runtime = Runtime(
        model_path=model_name,
        tokenizer_path=model_name,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
    )
    set_default_backend(runtime)

    @function
    def stream_chat(s, prompt_text: str):
        s += prompt_text
        s += gen("", max_new_tokens=max_new_tokens, stream=True, temperature=0.0)

    samples: List[Tuple[float, float, int]] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        first: Optional[float] = None
        chunks: List[str] = []
        for delta in stream_chat.run_stream(prompt_text=prompt, temperature=0.0):
            if delta and first is None:
                first = time.perf_counter()
            chunks.append(delta)
        t1 = time.perf_counter()
        tokens = len(runtime.tokenizer.encode("".join(chunks), add_special_tokens=False))
        samples.append(((first or t1) - t0, t1 - t0, tokens))
    ttfb_p50 = _p50([s[0] for s in samples])
    tp_p50 = _p50([s[2] / s[1] for s in samples if s[1] > 0])
    return ttfb_p50, tp_p50


def _run_sglang_vs_hf() -> None:
    compare = os.getenv("BENCH_INCLUDE_SGLANG", "0") == "1"
    if not compare:
        return
    prompt = "Hello from sglang vs HF!"
    max_new_tokens = 256
    repeat = max(1, int(os.getenv("BENCH_REPEAT", "3")))
    tp_size = 1
    dtype = _auto_dtype()
    logging.getLogger(__name__).info(
        "Running sglang vs HF | model=%s prompt_len=%d max_new_tokens=%d repeat=%d dtype=%s tp=%d",
        MODEL_NAME,
        len(prompt),
        max_new_tokens,
        repeat,
        dtype,
        tp_size,
    )
    hf_ttfb, hf_tp = _bench_hf_stream(
        model_name=MODEL_NAME,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        repeat=repeat,
        dtype=dtype,
    )
    sgl_ttfb, sgl_tp = _bench_sglang_runtime(
        model_name=MODEL_NAME,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        repeat=repeat,
        tensor_parallel_size=tp_size,
    )
    print("\nSGLang Runtime vs HF (p50):")
    print(f"- model={MODEL_NAME} prompt={prompt!r} max_new_tokens={max_new_tokens} repeat={repeat}")
    print(f"- HF       TTFB={hf_ttfb:.3f}s throughput={hf_tp:.2f} tok/s (dtype={dtype or 'auto'})")
    print(f"- sglang   TTFB={sgl_ttfb:.3f}s throughput={sgl_tp:.2f} tok/s (tp_size={tp_size})")


def main() -> None:
    repeat = max(1, int(os.getenv("BENCH_REPEAT", "3")))
    warmup = max(0, int(os.getenv("BENCH_WARMUP", "1")))
    concurrency = max(1, int(os.getenv("BENCH_CONCURRENCY", "4")))
    mixed_prompts = os.getenv("BENCH_MIXED_PROMPTS", "1") != "0"
    log_level = os.getenv("BENCH_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.WARNING),
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )

    backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
    engine = SGLangMiniEngine(
        backend=backend, max_new_tokens_default=max(w.max_new_tokens for w in _default_workloads())
    )

    _run_suite(
        backend=backend,
        engine=engine,
        workloads=_default_workloads(),
        repeat=repeat,
        warmup=warmup,
        concurrency=concurrency,
        mixed_prompts=mixed_prompts,
    )
    _run_sglang_vs_hf()


if __name__ == "__main__":
    main()
