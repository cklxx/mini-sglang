"""FastAPI server exposing a streaming /generate endpoint.

This mirrors sglang's API layer in miniature: a single POST /generate that
drives the engine and streams tokens back to the caller.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from io import BytesIO
from queue import SimpleQueue
from threading import Lock, Thread
from typing import Any, Callable, Generator, Optional, Protocol

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.diffusion.diffusers_image import DiffusersImageConfig, DiffusersImageRunner
from backend.diffusion.diffusers_video import DiffusersVideoConfig, DiffusersVideoRunner
from backend.diffusion.z_image import (
    DEFAULT_MODEL_ID as DEFAULT_Z_IMAGE_MODEL_ID,
)
from backend.diffusion.z_image import (
    ZImageConfig,
    ZImageRunner,
    parse_size,
    resolve_size,
)
from backend.factory import create_backend, resolve_backend_impl
from backend.hf.baseline import HFBaseline
from backend.hf.vlm import HFVLMRunner
from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME
from engine.engine import SGLangMiniEngine
from ipc.zmq_control import start_control_server
from multi_device import EnginePool
from utils.image_loader import extract_openai_image_url, load_pil_image


class VLMRunner(Protocol):
    def tokenize(self, text: str) -> list[int]: ...

    def generate_streaming(
        self,
        *,
        prompt: str,
        images: list[Any],
        max_new_tokens: int,
        log_stride: int = 32,
        stream_callback: Callable[[str], None] | None = None,
    ) -> tuple[str, float]: ...


logger = logging.getLogger(__name__)

app = FastAPI(title="sglang-mini")

STREAM_LOG_STRIDE = max(1, int(os.getenv("SERVER_STREAM_LOG_STRIDE", "32")))
_pool: EnginePool | None = None
_pool_lock = Lock()
_hf_runner: HFBaseline | None = None
_hf_runner_lock = Lock()
_vlm_runners: dict[tuple[str, str, str], VLMRunner] = {}
_vlm_runner_lock = Lock()
_z_image_runners: dict[ZImageConfig, ZImageRunner] = {}
_z_image_runner_lock = Lock()
_diffusers_image_runners: dict[DiffusersImageConfig, DiffusersImageRunner] = {}
_diffusers_image_runner_lock = Lock()
_diffusers_video_runners: dict[DiffusersVideoConfig, DiffusersVideoRunner] = {}
_diffusers_video_runner_lock = Lock()
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


def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    """Flatten OpenAI-style chat messages into a plain text prompt."""
    parts: list[str] = []
    for msg in messages:
        text_segments: list[str] = []
        if isinstance(msg.content, str):
            text_segments.append(msg.content)
        elif isinstance(msg.content, list):
            for segment in msg.content:
                if not isinstance(segment, dict):
                    continue
                if segment.get("type") == "text":
                    content = segment.get("text")
                    if isinstance(content, str):
                        text_segments.append(content)
        if text_segments:
            parts.append(f"{msg.role}: {' '.join(text_segments)}")
    return "\n".join(parts)


def _extract_chat_inputs(messages: list[ChatMessage]) -> tuple[str, list[str]]:
    """Return (prompt_text, image_urls) from OpenAI-style messages."""

    image_token = os.getenv("VLM_IMAGE_TOKEN", "<image>")
    parts: list[str] = []
    image_urls: list[str] = []
    for msg in messages:
        segments: list[str] = []
        if isinstance(msg.content, str):
            if msg.content:
                segments.append(msg.content)
        elif isinstance(msg.content, list):
            for segment in msg.content:
                if not isinstance(segment, dict):
                    continue
                seg_type = str(segment.get("type") or "")
                if seg_type == "text":
                    text = segment.get("text")
                    if isinstance(text, str) and text:
                        segments.append(text)
                elif seg_type in {"image_url", "image", "input_image"}:
                    url = extract_openai_image_url(segment)
                    if url:
                        image_urls.append(url)
                        segments.append(image_token)
        if segments:
            parts.append(f"{msg.role}: {' '.join(segments)}")
    return "\n".join(parts), image_urls


def _chat_delta_chunk(
    *,
    model: str,
    content: Optional[str] = None,
    finish_reason: Optional[str] = None,
    usage: Optional[dict[str, int]] = None,
) -> bytes:
    payload = {
        "id": f"chatcmpl-{int(time.time() * 1000)}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _ensure_hf_runner(device: str) -> HFBaseline:
    """Lazily construct a standalone HF baseline runner (thread-safe)."""
    global _hf_runner
    with _hf_runner_lock:
        if _hf_runner is None or _hf_runner.device != device:
            _hf_runner = HFBaseline(model_name=MODEL_NAME, device=device)
    return _hf_runner


def _mlx_vlm_available() -> bool:
    try:
        import mlx.core as _mx  # noqa: F401
    except Exception:
        return False
    try:
        import mlx_vlm as _mlx_vlm  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_vlm_backend(device: str) -> str:
    backend = os.getenv("VLM_BACKEND", "auto").lower()
    if backend not in {"auto", "hf", "mlx"}:
        logger.warning("Unrecognized VLM_BACKEND=%s; defaulting to auto", backend)
        backend = "auto"
    if backend == "hf":
        return "hf"
    if backend == "mlx":
        if device != "mps":
            logger.warning("VLM_BACKEND=mlx requires device=mps; falling back to hf")
            return "hf"
        return "mlx"
    if device == "mps" and _mlx_vlm_available():
        return "mlx"
    return "hf"


def _ensure_vlm_runner(backend: str, model_name: str, device: str) -> VLMRunner:
    """Lazily construct a VLM runner keyed by (backend, model, device)."""

    key = (backend, model_name, device)
    with _vlm_runner_lock:
        runner = _vlm_runners.get(key)
        if runner is None:
            if _vlm_runners:
                # Keep memory bounded; VLM weights are large.
                _vlm_runners.clear()
            if backend == "mlx":
                from backend.mlx.vlm import MlxVLMRunner

                runner = MlxVLMRunner(model_name=model_name, device=device)
            else:
                runner = HFVLMRunner(model_name=model_name, device=device)
            _vlm_runners[key] = runner
        return runner


def _ensure_z_image_runner(config: ZImageConfig) -> ZImageRunner:
    with _z_image_runner_lock:
        runner = _z_image_runners.get(config)
        if runner is None:
            if _z_image_runners:
                _z_image_runners.clear()
            runner = ZImageRunner(config=config)
            _z_image_runners[config] = runner
        return runner


def _ensure_diffusers_image_runner(config: DiffusersImageConfig) -> DiffusersImageRunner:
    with _diffusers_image_runner_lock:
        runner = _diffusers_image_runners.get(config)
        if runner is None:
            if _diffusers_image_runners:
                _diffusers_image_runners.clear()
            runner = DiffusersImageRunner(config=config)
            _diffusers_image_runners[config] = runner
        return runner


def _ensure_diffusers_video_runner(config: DiffusersVideoConfig) -> DiffusersVideoRunner:
    with _diffusers_video_runner_lock:
        runner = _diffusers_video_runners.get(config)
        if runner is None:
            if _diffusers_video_runners:
                _diffusers_video_runners.clear()
            runner = DiffusersVideoRunner(config=config)
            _diffusers_video_runners[config] = runner
        return runner


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


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, object]]


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage]
    model: Optional[str] = None
    max_tokens: Optional[int] = Field(None, ge=1)
    temperature: float = 0.0
    stream: Optional[bool] = True
    stream_options: Optional[dict[str, object]] = None
    ignore_eos: Optional[bool] = None


class ImagesGenerationRequest(BaseModel):
    prompt: str
    backend: str = Field("z_image", description="z_image | diffusers")
    model: Optional[str] = None
    n: int = Field(1, ge=1, le=8)
    size: Optional[str] = Field(None, description="WxH, e.g. 1024x1024")
    response_format: str = Field("b64_json", description="b64_json only")
    seed: Optional[int] = None
    steps: int = Field(8, ge=1, le=100)
    guidance_scale: float = Field(0.0, ge=0.0)
    device: str = Field("auto", description="auto|mps|cuda|cpu")
    attention_backend: str = Field("sdpa", description="sdpa|flash2|flash3")
    compile: bool = False
    cpu_offload: bool = False
    gguf: Optional[str] = None
    gguf_file: Optional[str] = None
    model_dir: Optional[str] = None
    aspect: Optional[str] = Field(None, description="1:1|16:9|9:16|4:3|3:4")
    height: int = Field(1024, ge=1)
    width: int = Field(1024, ge=1)


class VideosGenerationRequest(BaseModel):
    prompt: Optional[str] = None
    image_url: Optional[str] = None
    model: Optional[str] = None
    n: int = Field(1, ge=1, le=4)
    size: Optional[str] = Field(None, description="WxH, e.g. 1024x576")
    response_format: str = Field("frames_b64_png", description="frames_b64_png | b64_mp4")
    seed: Optional[int] = None
    steps: int = Field(25, ge=1, le=100)
    guidance_scale: float = Field(0.0, ge=0.0)
    fps: int = Field(8, ge=1, le=30)
    num_frames: int = Field(14, ge=1, le=120)
    device: str = Field("auto", description="auto|mps|cuda|cpu")
    cpu_offload: bool = False
    model_dir: Optional[str] = None
    aspect: Optional[str] = Field(None, description="1:1|16:9|9:16|4:3|3:4")
    height: int = Field(576, ge=1)
    width: int = Field(1024, ge=1)


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    prompt, image_urls = _extract_chat_inputs(request.messages)
    if not prompt:
        raise HTTPException(status_code=400, detail="messages must include text content")

    if image_urls:
        timeout_s = float(os.getenv("VLM_IMAGE_HTTP_TIMEOUT_S", "5"))
        max_bytes = int(os.getenv("VLM_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
        allow_file = os.getenv("ALLOW_FILE_IMAGE_URL", "0") != "0"
        try:
            images = [
                load_pil_image(
                    url, timeout_s=timeout_s, max_bytes=max_bytes, allow_file=allow_file
                )
                for url in image_urls
            ]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image_url: {exc}") from exc

        pool = _ensure_pool()
        prompt_ids = pool.primary_backend.tokenize(prompt)
        engine, lease = pool.pick(prompt_ids=prompt_ids)
        device = engine.backend.device
        max_tokens = pool.adapt_max_new_tokens(request.max_tokens or MAX_NEW_TOKENS_DEFAULT)
        vlm_model_name = request.model or os.getenv("VLM_MODEL_NAME") or MODEL_NAME
        vlm_backend = _resolve_vlm_backend(device)
        try:
            runner = _ensure_vlm_runner(vlm_backend, vlm_model_name, device)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"VLM backend unavailable (backend={vlm_backend}): {exc}",
            ) from exc

        def count_tokens(text: str) -> int:
            tokenize = getattr(runner, "tokenize", None)
            if not callable(tokenize):
                return 0
            try:
                return len(tokenize(text))
            except Exception:
                return 0

        if request.stream is False:
            start_time = time.perf_counter()
            generated_text = ""
            try:
                generated_text, _ = runner.generate_streaming(
                    prompt=prompt,
                    images=images,
                    max_new_tokens=max_tokens,
                    stream_callback=None,
                )
                return JSONResponse(
                    {
                        "id": f"chatcmpl-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "created": int(time.time()),
                        "model": vlm_model_name,
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": generated_text},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {
                            "prompt_tokens": count_tokens(prompt) or len(prompt_ids),
                            "completion_tokens": count_tokens(generated_text),
                            "total_tokens": (count_tokens(prompt) or len(prompt_ids))
                            + count_tokens(generated_text),
                        },
                    }
                )
            finally:
                duration = time.perf_counter() - start_time
                total_tokens = (count_tokens(prompt) or len(prompt_ids)) + count_tokens(
                    generated_text
                )
                pool.record_generation(duration_s=duration, tokens=total_tokens)
                pool.release(lease)

        start_time = time.perf_counter()
        queue_vlm: SimpleQueue[bytes] = SimpleQueue()
        sentinel_vlm = b"__mini_sglang_done__"

        def stream_callback_vlm(delta: str) -> None:
            if delta:
                queue_vlm.put(_chat_delta_chunk(model=vlm_model_name, content=delta))

        def run_vlm() -> None:
            generated_text = ""
            try:
                generated_text, _ = runner.generate_streaming(
                    prompt=prompt,
                    images=images,
                    max_new_tokens=max_tokens,
                    stream_callback=stream_callback_vlm,
                )
                prompt_tokens = count_tokens(prompt) or len(prompt_ids)
                completion_tokens = count_tokens(generated_text)
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                }
                queue_vlm.put(
                    _chat_delta_chunk(model=vlm_model_name, finish_reason="stop", usage=usage)
                )
            except Exception as exc:  # pragma: no cover - streaming error surface
                logger.exception("VLM chat completion stream failed: %s", exc)
                err_payload = {"error": {"message": str(exc)}}
                queue_vlm.put(
                    f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n".encode("utf-8")
                )
            finally:
                duration = time.perf_counter() - start_time
                total_tokens = (count_tokens(prompt) or len(prompt_ids)) + count_tokens(
                    generated_text
                )
                pool.record_generation(duration_s=duration, tokens=total_tokens)
                pool.release(lease)
                queue_vlm.put(b"data: [DONE]\n\n")
                queue_vlm.put(sentinel_vlm)

        Thread(target=run_vlm, daemon=True).start()

        def event_stream_vlm() -> Generator[bytes, None, None]:
            while True:
                item = queue_vlm.get()
                if item == sentinel_vlm:
                    break
                yield item

        return StreamingResponse(event_stream_vlm(), media_type="text/event-stream")

    model_name = MODEL_NAME
    if request.stream is False:
        pool = _ensure_pool()
        prompt_ids = pool.primary_backend.tokenize(prompt)
        engine, lease = pool.pick(prompt_ids=prompt_ids)
        max_tokens = pool.adapt_max_new_tokens(request.max_tokens or MAX_NEW_TOKENS_DEFAULT)
        generated_text = ""
        start_time = time.perf_counter()
        try:
            generated_text = engine.run_generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                stream_callback=lambda _: None,
                prompt_ids=prompt_ids,
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
            pool.record_generation(duration_s=duration, tokens=total_tokens)
            pool.release(lease)

        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": gen_tokens,
            "total_tokens": total_tokens,
        }
        response = {
            "id": f"chatcmpl-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage,
        }
        return JSONResponse(response)

    start_time = time.perf_counter()
    queue_engine: SimpleQueue[bytes] = SimpleQueue()
    sentinel_engine = b"__mini_sglang_done__"

    def stream_callback_engine(delta: str) -> None:
        if delta:
            queue_engine.put(_chat_delta_chunk(model=model_name, content=delta))

    def run_engine() -> None:
        local_pool = _ensure_pool()
        prompt_ids = local_pool.primary_backend.tokenize(prompt)
        if local_pool.async_prefill_enabled:
            local_pool.enqueue_prefill(prompt)
        engine, lease = local_pool.pick(prompt_ids=prompt_ids)
        max_tokens = local_pool.adapt_max_new_tokens(request.max_tokens or MAX_NEW_TOKENS_DEFAULT)
        generated_text: str = ""
        try:
            generated_text = engine.run_generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                stream_callback=stream_callback_engine,
                prompt_ids=prompt_ids,
            )
            completion_tokens = (
                len(engine.backend.tokenize(generated_text)) if generated_text else 0
            )
            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_ids) + completion_tokens,
            }
            queue_engine.put(_chat_delta_chunk(model=model_name, finish_reason="stop", usage=usage))
        except Exception as exc:  # pragma: no cover - streaming error surface
            logger.exception("Chat completion stream failed: %s", exc)
            err_payload = {"error": {"message": str(exc)}}
            queue_engine.put(
                f"data: {json.dumps(err_payload, ensure_ascii=False)}\n\n".encode("utf-8")
            )
        finally:
            duration = time.perf_counter() - start_time
            total_tokens = len(prompt_ids)
            try:
                gen_tokens = len(engine.backend.tokenize(generated_text)) if generated_text else 0
                total_tokens += gen_tokens
            except Exception:
                gen_tokens = 0
            local_pool.record_generation(duration_s=duration, tokens=total_tokens)
            local_pool.release(lease)
            queue_engine.put(b"data: [DONE]\n\n")
            queue_engine.put(sentinel_engine)

    Thread(target=run_engine, daemon=True).start()

    def event_stream_engine() -> Generator[bytes, None, None]:
        while True:
            item = queue_engine.get()
            if item == sentinel_engine:
                break
            yield item

    return StreamingResponse(event_stream_engine(), media_type="text/event-stream")


@app.post("/v1/images/generations")
def image_generations(request: ImagesGenerationRequest):
    if request.response_format != "b64_json":
        raise HTTPException(status_code=400, detail="only response_format=b64_json is supported")

    try:
        if request.size:
            height, width = parse_size(request.size)
        else:
            height, width = resolve_size(request.aspect, request.height, request.width)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    backend = request.backend.lower().strip()
    if backend not in {"z_image", "diffusers"}:
        raise HTTPException(status_code=400, detail="backend must be 'z_image' or 'diffusers'")

    runner: DiffusersImageRunner | ZImageRunner
    if backend == "diffusers":
        model_id = request.model or os.getenv("IMAGE_MODEL_ID")
        if not model_id:
            raise HTTPException(status_code=400, detail="model is required for backend=diffusers")
        if request.gguf or request.gguf_file:
            raise HTTPException(
                status_code=400, detail="gguf is only supported for backend=z_image"
            )
        diffusers_config = DiffusersImageConfig(
            model_id=model_id,
            model_dir=request.model_dir,
            device=request.device,
            cpu_offload=request.cpu_offload,
        )
        runner = _ensure_diffusers_image_runner(diffusers_config)
    else:
        model_id = request.model or DEFAULT_Z_IMAGE_MODEL_ID
        z_image_config = ZImageConfig(
            model_id=model_id,
            model_dir=request.model_dir,
            device=request.device,
            attention_backend=request.attention_backend,
            compile_transformer=request.compile,
            cpu_offload=request.cpu_offload,
            gguf=request.gguf,
            gguf_file=request.gguf_file,
        )
        runner = _ensure_z_image_runner(z_image_config)

    logger.info(
        "Incoming image generation | model=%s n=%d steps=%d size=%dx%d",
        model_id,
        request.n,
        request.steps,
        width,
        height,
    )
    try:
        images = runner.generate(
            prompt=request.prompt,
            num_images=request.n,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=height,
            width=width,
            seed=request.seed,
        )
    except Exception as exc:  # pragma: no cover - surface as 500
        logger.exception("Image generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    data: list[dict[str, str]] = []
    for img in images:
        buf = BytesIO()
        img.save(buf, format="PNG")
        data.append({"b64_json": base64.b64encode(buf.getvalue()).decode("ascii")})

    return JSONResponse({"created": int(time.time()), "data": data})


@app.post("/v1/videos/generations")
def video_generations(request: VideosGenerationRequest):
    if request.response_format not in {"frames_b64_png", "b64_mp4"}:
        raise HTTPException(
            status_code=400, detail="response_format must be 'frames_b64_png' or 'b64_mp4'"
        )

    try:
        if request.size:
            height, width = parse_size(request.size)
        else:
            height, width = resolve_size(request.aspect, request.height, request.width)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    image = None
    if request.image_url:
        timeout_s = float(os.getenv("VLM_IMAGE_HTTP_TIMEOUT_S", "5"))
        max_bytes = int(os.getenv("VLM_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
        allow_file = os.getenv("ALLOW_FILE_IMAGE_URL", "0") != "0"
        try:
            image = load_pil_image(
                request.image_url, timeout_s=timeout_s, max_bytes=max_bytes, allow_file=allow_file
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid image_url: {exc}") from exc

    model_id = request.model or os.getenv("VIDEO_MODEL_ID")
    if not model_id:
        raise HTTPException(status_code=400, detail="model is required (or set VIDEO_MODEL_ID)")

    config = DiffusersVideoConfig(
        model_id=model_id,
        model_dir=request.model_dir,
        device=request.device,
        cpu_offload=request.cpu_offload,
    )
    runner = _ensure_diffusers_video_runner(config)

    logger.info(
        "Incoming video generation | model=%s n=%d steps=%d frames=%d fps=%d size=%dx%d",
        model_id,
        request.n,
        request.steps,
        request.num_frames,
        request.fps,
        width,
        height,
    )
    try:
        videos = runner.generate(
            prompt=request.prompt,
            image=image,
            num_videos=request.n,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=height,
            width=width,
            num_frames=request.num_frames,
            fps=request.fps,
            seed=request.seed,
        )
    except Exception as exc:  # pragma: no cover - surface as 500
        logger.exception("Video generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if request.response_format == "b64_mp4":
        try:
            import imageio.v3 as iio  # type: ignore[import-not-found]
            import numpy as np  # type: ignore[import-not-found]
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="b64_mp4 requires `imageio` and `numpy` installed.",
            ) from exc

        data_mp4: list[dict[str, object]] = []
        for frames in videos:
            arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
            buf = BytesIO()
            try:
                iio.imwrite(buf, arrays, fps=request.fps, format_hint=".mp4")
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"mp4 encoding failed: {exc}") from exc
            data_mp4.append(
                {"b64_mp4": base64.b64encode(buf.getvalue()).decode("ascii"), "fps": request.fps}
            )
        return JSONResponse({"created": int(time.time()), "data": data_mp4})

    data: list[dict[str, object]] = []
    for frames in videos:
        frame_b64: list[str] = []
        for frame in frames:
            buf = BytesIO()
            frame.save(buf, format="PNG")
            frame_b64.append(base64.b64encode(buf.getvalue()).decode("ascii"))
        data.append({"frames_b64_png": frame_b64, "fps": request.fps})

    return JSONResponse({"created": int(time.time()), "data": data})


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
                    if hasattr(engine.backend, "generate_streaming_baseline"):
                        generated_text, _ = engine.backend.generate_streaming_baseline(
                            prompt_ids=prompt_ids,
                            max_new_tokens=max_tokens or MAX_NEW_TOKENS_DEFAULT,
                            stream_callback=stream_callback,
                        )
                    else:
                        hf_runner = _ensure_hf_runner(engine.backend.device)
                        generated_text, _ = hf_runner.generate_streaming(
                            prompt=request.prompt,
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
