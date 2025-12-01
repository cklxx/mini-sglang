"""FastAPI server exposing a streaming /generate endpoint.

This mirrors sglang's API layer in miniature: a single POST /generate that
drives the engine and streams tokens back to the caller.
"""
from __future__ import annotations

import json
import logging
from queue import SimpleQueue
from threading import Thread
from typing import Any, Dict, Generator, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.model_backend import ModelBackend
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from engine.engine import SGLangMiniEngine
from optimizations import warmup_engine

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="sglang-mini")

backend = ModelBackend(model_name=MODEL_NAME, device=get_device())
engine = SGLangMiniEngine(backend=backend, max_new_tokens_default=MAX_NEW_TOKENS_DEFAULT)
STREAM_LOG_STRIDE = max(1, int(os.getenv("SERVER_STREAM_LOG_STRIDE", "32")))


@app.on_event("startup")
def _warm_server() -> None:
    warm_tokens = min(16, MAX_NEW_TOKENS_DEFAULT)
    logger.info("Server warmup starting | tokens=%d model=%s", warm_tokens, MODEL_NAME)
    warmup_engine(engine, max_new_tokens=warm_tokens)
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
                "HTTP API is streaming-only. Use benchmark.py or one_click_compare.py "
                "to compare against vanilla generate()."
            ),
        )

    mode = (request.mode or "sglang").lower()
    if mode not in ("sglang", "hf"):
        raise HTTPException(status_code=400, detail="mode must be 'sglang' or 'hf'")

    def event_stream() -> Generator[bytes, None, None]:
        queue: SimpleQueue[object] = SimpleQueue()
        sentinel = object()

        def stream_callback(text_delta: str) -> None:
            chunk: Dict[str, Any] = {"text_delta": text_delta}
            queue.put((json.dumps(chunk) + "\n").encode("utf-8"))
            if len(chunk.get("text_delta", "")) > 0:
                stream_callback.count = getattr(stream_callback, "count", 0) + 1
                if stream_callback.count <= 2 or stream_callback.count % STREAM_LOG_STRIDE == 0:
                    logger.info(
                        "Streamed chunk %03d (mode=%s): %r",
                        stream_callback.count,
                        mode,
                        chunk,
                    )

        def run_engine() -> None:
            if mode == "sglang":
                engine.run_generate(
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    stream_callback=stream_callback,
                )
            else:
                prompt_ids = backend.tokenize(request.prompt)
                backend.generate_streaming_baseline(
                    prompt_ids=prompt_ids,
                    max_new_tokens=request.max_new_tokens or MAX_NEW_TOKENS_DEFAULT,
                    stream_callback=stream_callback,
                )
            queue.put(json.dumps({"event": "done"}).encode("utf-8") + b"\n")
            queue.put(sentinel)
            logger.info(
                "Generation thread completed for prompt length=%d mode=%s", len(request.prompt), mode
            )

        thread = Thread(target=run_engine, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is sentinel:
                break
            yield item  # type: ignore[misc]

    return StreamingResponse(event_stream(), media_type="text/event-stream")
