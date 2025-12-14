"""HF-based VLM runner (best-effort, streaming).

This is intentionally a baseline implementation using `TextIteratorStreamer`.
It is used by the FastAPI server when chat messages contain `image_url` segments.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Callable, Optional, Tuple, cast

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, TextIteratorStreamer

from utils.runtime import configure_torch, inference_context

logger = logging.getLogger(__name__)


try:  # pragma: no cover - depends on transformers version
    from transformers import AutoModelForVision2Seq  # type: ignore
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None  # type: ignore[assignment]


def _resolve_model_path(model_name: str) -> str:
    override = os.getenv("MODEL_LOCAL_DIR")
    if override:
        logger.info("HFVLMRunner: Using MODEL_LOCAL_DIR=%s", override)
        return override
    return model_name


def _device_type(device: str) -> str:
    if device.startswith("cuda"):
        return "cuda"
    if device == "mps":
        return "mps"
    return "cpu"


def _preferred_dtype(device: str) -> torch.dtype:
    dtype = torch.float32
    device_kind = _device_type(device)
    if device_kind == "cuda":
        if torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    elif device_kind == "mps":
        dtype = torch.float16
    return dtype


class HFVLMRunner:
    """Minimal VLM streaming helper on top of transformers `generate()`."""

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        device_kind = _device_type(device)
        configure_torch(device_kind)

        model_path = _resolve_model_path(model_name)
        trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "0") != "0"
        desired_dtype = _preferred_dtype(device)

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
        self.tokenizer = cast(Any, tokenizer)

        model: Any | None = None
        if AutoModelForVision2Seq is not None:
            try:
                model = AutoModelForVision2Seq.from_pretrained(
                    model_path,
                    trust_remote_code=trust_remote_code,
                    torch_dtype=desired_dtype,
                )
            except Exception as exc:
                logger.info("AutoModelForVision2Seq load failed; falling back (%s)", exc)
                model = None

        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=desired_dtype,
            )

        self.model = cast(Any, model)
        self.model.to(torch.device(self.device))
        self.model.eval()

        self._lock = threading.Lock()

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        self.eos_token_id = eos_token_id if eos_token_id is not None else pad_token_id
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            cast(Any, self.tokenizer).pad_token_id = self.eos_token_id or 0

    def tokenize(self, text: str) -> list[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            try:
                return list(self.tokenizer.encode(text))
            except Exception:
                return []

    def generate_streaming(
        self,
        *,
        prompt: str,
        images: list[Any],
        max_new_tokens: int,
        log_stride: int = 32,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, float]:
        if not images:
            raise ValueError("VLM runner requires at least one image input")

        try:
            model_inputs = self.processor(
                prompt,
                images=images,
                return_tensors="pt",
                padding=True,
            )
        except TypeError:
            model_inputs = self.processor(
                prompt,
                images[0] if len(images) == 1 else images,
                return_tensors="pt",
                padding=True,
            )

        inputs: dict[str, Any] = {}
        for k, v in cast(Any, model_inputs).items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
            else:
                inputs[k] = v

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        log_stride = max(1, int(os.getenv("VLM_STREAM_LOG_STRIDE", str(log_stride))))

        def _generate() -> None:
            auto_ctx, inf_ctx = inference_context(_device_type(self.device))
            with auto_ctx, inf_ctx:
                self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=self.eos_token_id,
                    streamer=streamer,
                    top_p=None,
                    top_k=None,
                    temperature=None,
                )

        with self._lock:
            start = time.perf_counter()
            thread = threading.Thread(target=_generate, daemon=True)
            thread.start()

            chunks: list[str] = []
            for idx, token_text in enumerate(streamer, 1):
                if stream_callback:
                    stream_callback(token_text)
                chunks.append(token_text)
                if idx == 1 or idx % log_stride == 0:
                    logger.info(
                        "[vlm] chunk %03d: %r (log_stride=%d)", idx, token_text, log_stride
                    )

            thread.join()
            duration = time.perf_counter() - start

        full_text = "".join(chunks)
        logger.info("VLM complete | chunks=%d", len(chunks))
        return full_text, duration
