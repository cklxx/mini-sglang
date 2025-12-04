"""FastAPI server exposing a streaming /generate endpoint.

This mirrors sglang's API layer in miniature: a single POST /generate that
drives the engine and streams tokens back to the caller.
"""

from __future__ import annotations

import json
import logging
import os
import time
from queue import SimpleQueue
from threading import Lock, Thread
from typing import Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.backend_factory import create_backend, resolve_backend_impl
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME
from engine.engine import SGLangMiniEngine
from ipc.zmq_control import start_control_server
from multi_device import EnginePool

logger = logging.getLogger(__name__)

app = FastAPI(title="sglang-mini")

STREAM_LOG_STRIDE = max(1, int(os.getenv("SERVER_STREAM_LOG_STRIDE", "32")))
_pool: EnginePool | None = None
_pool_lock = Lock()
_logging_configured = False


def _configure_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
    )
    _logging_configured = True


def _ensure_pool() -> EnginePool:
    global _pool
    if _pool is not None:
        return _pool
    with _pool_lock:
        if _pool is None:
            _pool = EnginePool(
                backend_ctor=create_backend,
                SGLangMiniEngine=SGLangMiniEngine,
                model_name=MODEL_NAME,
                max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT,
            )
    return _pool


def get_pool() -> EnginePool:
    """Public accessor for the lazily initialized EnginePool."""
    return _ensure_pool()


@app.on_event("startup")
def _warm_server() -> None:
    _configure_logging()
    pool = _ensure_pool()
    backend_impl = resolve_backend_impl(pool.primary_backend.device)
    logger.info("Server using backend=%s device=%s", backend_impl, pool.primary_backend.device)
    warm_tokens = min(16, MAX_NEW_TOKENS_DEFAULT)
    logger.info("Server warmup starting | tokens=%d model=%s", warm_tokens, MODEL_NAME)
    pool.warm(warm_tokens)
    warm_prefixes = os.getenv("WARM_PREFIXES")
    if warm_prefixes:
        prompts = [p for p in warm_prefixes.split("||") if p.strip()]
        if prompts:
            logger.info("Warming %d prefixes into cache", len(prompts))
            pool.warm_prefixes(prompts)
    if os.getenv("ZMQ_CONTROL", "0") != "0":
        endpoint = os.getenv("ZMQ_CONTROL_ENDPOINT", "tcp://127.0.0.1:5557")
        logger.info("Starting ZMQ control server on %s", endpoint)
        start_control_server(
            endpoint=endpoint,
            handler=lambda req: _handle_control(req),
        )
    logger.info("Server warmup done")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = Field(None, ge=1)
    stream: Optional[bool] = True
    mode: Optional[str] = Field("sglang", description="sglang | hf")


@app.post("/generate")
def generate(request: GenerateRequest):
    logger.info(
        "Incoming /generate request | prompt_preview=%r max_new_tokens=%s",
        request.prompt[:50],
        request.max_new_tokens,
    )

    if request.stream is False:
        logger.info(
            "Non-streaming HTTP mode is disabled; use benchmark/one-click scripts for baselines",
        )
        raise HTTPException(
            status_code=400,
            detail=(
                "HTTP API is streaming-only. Use bench_suite.py to compare "
                "against vanilla generate()."
            ),
        )

    mode = (request.mode or "sglang").lower()
    if mode not in ("sglang", "hf"):
        raise HTTPException(status_code=400, detail="mode must be 'sglang' or 'hf'")

    def event_stream() -> Generator[bytes, None, None]:
        queue: SimpleQueue[bytes] = SimpleQueue()
        sentinel = b"__mini_sglang_done__"
        send_count = 0
        start_time = time.perf_counter()

        def stream_callback(text_delta: str) -> None:
            nonlocal send_count
            chunk: dict[str, str] = {"text_delta": text_delta}
            queue.put((json.dumps(chunk) + "\n").encode("utf-8"))
            if text_delta:
                send_count += 1
                if send_count <= 2 or send_count % STREAM_LOG_STRIDE == 0:
                    logger.info("Streamed chunk %03d (mode=%s): %r", send_count, mode, chunk)

        def run_engine() -> None:
            local_pool = _ensure_pool()
            prompt_ids = local_pool.primary_backend.tokenize(request.prompt)
            if local_pool.async_prefill_enabled:
                local_pool.enqueue_prefill(request.prompt)
            requested_tokens = request.max_new_tokens or MAX_NEW_TOKENS_DEFAULT
            engine, lease = local_pool.pick(prompt_ids=prompt_ids)
            max_tokens = local_pool.adapt_max_new_tokens(requested_tokens)
            generated_text: str = ""
            try:
                if mode == "sglang":
                    generated_text = engine.run_generate(
                        prompt=request.prompt,
                        max_new_tokens=max_tokens,
                        stream_callback=stream_callback,
                        prompt_ids=prompt_ids,
                    )
                else:
                    generated_text, _ = engine.backend.generate_streaming_baseline(
                        prompt_ids=prompt_ids,
                        max_new_tokens=max_tokens or MAX_NEW_TOKENS_DEFAULT,
                        stream_callback=stream_callback,
                    )
                queue.put(json.dumps({"event": "done"}).encode("utf-8") + b"\n")
                queue.put(sentinel)
                logger.info(
                    "Generation thread completed | prompt_len=%d mode=%s cache=%s",
                    len(request.prompt),
                    mode,
                    engine.backend.cache_metrics(),
                )
            finally:
                duration = time.perf_counter() - start_time
                try:
                    gen_tokens = (
                        len(engine.backend.tokenize(generated_text)) if generated_text else 0
                    )
                except Exception:
                    gen_tokens = 0
                total_tokens = len(prompt_ids) + gen_tokens
                local_pool.record_generation(duration_s=duration, tokens=total_tokens)
                local_pool.release(lease)

        thread = Thread(target=run_engine, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item == sentinel:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/metrics")
def metrics():
    """Lightweight JSON metrics for cache hits/misses and inflight counts."""
    pool = _ensure_pool()
    return JSONResponse(pool.metrics())


def _handle_control(req: dict[str, object]) -> dict[str, object]:
    raw_cmd = req.get("cmd") or ""
    cmd = str(raw_cmd).lower()
    if cmd == "metrics":
        return _ensure_pool().metrics()
    if cmd == "warm":
        tokens_obj = req.get("tokens", 8)
        if isinstance(tokens_obj, (int, float, str)):
            try:
                tokens = int(tokens_obj)
            except Exception:
                tokens = 8
        else:
            tokens = 8
        pool = _ensure_pool()
        pool.warm(tokens)
        return {"ok": True, "warmed_tokens": tokens}
    return {"error": "unknown_command"}
