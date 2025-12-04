"""Engine layer implementing prefill + decode with streaming callbacks.

This is the mini analogue of sglang's engine: it orchestrates per-request
state, prefill + decode loops, and streaming callbacks on top of the backend.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, Optional

import torch

from utils.runtime import inference_context
from utils.timer import Timer

logger = logging.getLogger(__name__)


class SGLangMiniEngine:
    """Simple engine orchestrating generation using a ModelBackend."""

    def __init__(
        self,
        backend: Any,
        max_new_tokens_default: int,
        decode_log_stride: Optional[int] = None,
    ) -> None:
        self.backend = backend
        self.max_new_tokens_default = max_new_tokens_default
        self.decode_log_stride = decode_log_stride or int(os.getenv("DECODE_LOG_STRIDE", "256"))

    def run_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        stream_callback: Callable[[str], None],
        prompt_ids: Optional[list[int]] = None,
    ) -> str:
        """Generate text using prefill + decode while invoking a stream callback."""

        # Normalize prompt/token budget up front for consistent logging and limits.
        fast_stream_decode = os.getenv("FAST_STREAM_DECODE", "1") != "0"
        prompt_tokens: list[int] = (
            prompt_ids if prompt_ids is not None else self.backend.tokenize(prompt)
        )
        requested_tokens = max_new_tokens or self.max_new_tokens_default
        max_tokens = self.backend.cap_max_new_tokens(len(prompt_tokens), requested_tokens)
        if max_tokens is None:
            max_tokens = requested_tokens

        # Give backends a chance to prepare cache or allocator state before the run.
        if hasattr(self.backend, "prepare_generation"):
            try:
                self.backend.prepare_generation(max_tokens)
            except Exception as exc:  # pragma: no cover - best-effort hint
                logger.debug("prepare_generation hint failed on backend: %s", exc)

        logger.info(
            "Starting generation run | prompt_tokens=%d max_new_tokens=%d",
            len(prompt_tokens),
            max_tokens,
        )

        generated_ids: list[int] = []
        kv_cache = None
        eos_token_id = self.backend.eos_token_id
        finished = False

        max_context_length = getattr(self.backend, "max_context_length", None)
        max_context_margin = getattr(self.backend, "max_context_margin", 0)
        auto_ctx, inf_ctx = inference_context(self.backend.device)
        text_chunks: list[str] = []
        timer = Timer()
        with auto_ctx, inf_ctx:
            # Prefill once to seed KV cache and stream the first token.
            timer.mark("prefill_start")
            first_token_id, kv_cache = self.backend.prefill_forward(
                prompt_tokens, use_context=False
            )
            if self.backend.device.startswith("cuda"):
                torch.cuda.synchronize()
            timer.mark("prefill_end")
            generated_ids.append(first_token_id)
            prefill_text = self.backend.decode_tokens([first_token_id])
            if not fast_stream_decode:
                text_chunks.append(prefill_text)
            logger.info("Prefill emitted token_id=%d text=%r", first_token_id, prefill_text)
            stream_callback(prefill_text)

            if first_token_id == eos_token_id or max_tokens == 1:
                finished = True
            else:
                # Decode token-by-token while streaming partial text.
                timer.mark("decode_start")
                for step_index in range(max_tokens - 1):
                    if finished:
                        break
                    if max_context_length is not None:
                        used = len(prompt_tokens) + len(generated_ids)
                        budget_left = max_context_length - used - max_context_margin
                        if budget_left <= 0:
                            logger.info(
                                "Stopping decode to avoid context overflow "
                                "(used=%d, max_ctx=%d, margin=%d)",
                                used,
                                max_context_length,
                                max_context_margin,
                            )
                            finished = True
                            break
                    last_token_id = generated_ids[-1]
                    step_start = time.perf_counter()
                    next_token_id, kv_cache = self.backend.decode_forward(
                        last_token_id, kv_cache, use_context=False
                    )
                    step_end = time.perf_counter()
                    generated_ids.append(next_token_id)
                    if next_token_id == eos_token_id:
                        finished = True
                    decode_delta = None
                    if not fast_stream_decode:
                        decode_delta = self.backend.decode_tokens([next_token_id])
                    should_log = (
                        step_index == 0
                        or (step_index + 1) % max(1, self.decode_log_stride) == 0
                        or finished
                    )
                    if should_log:
                        logger.info(
                            (
                                "Decode step %d emitted token_id=%d text=%r "
                                "finished=%s step_time=%.4fs"
                            ),
                            step_index,
                            next_token_id,
                            decode_delta if decode_delta is not None else "<skipped>",
                            finished,
                            step_end - step_start,
                        )
                    stream_callback(decode_delta if decode_delta is not None else " ")
                    if decode_delta is not None:
                        text_chunks.append(decode_delta)
                if self.backend.device.startswith("cuda"):
                    torch.cuda.synchronize()
                timer.mark("decode_end")

            # We already decoded per-step deltas; join them instead of re-decoding full ids.
            if fast_stream_decode:
                full_text = self.backend.decode_tokens(generated_ids)
            else:
                full_text = "".join(text_chunks)
            finished = True
        timer.mark("end")

        # Build timing/throughput stats after the generation run finishes.
        prefill_time = timer.span("prefill_start", "prefill_end")
        decode_time = timer.span("decode_start", "decode_end")
        total_time = timer.span("start", "end")
        decode_tokens = max(0, len(generated_ids) - 1)
        throughput = (len(generated_ids) / total_time) if total_time > 0 else 0.0
        message = (
            "Generation complete | tokens=%d prefill=%.3fs decode=%.3fs total=%.3fs "
            "decode_tokens=%d throughput=%.2f tok/s"
        )
        logger.info(
            message,
            len(generated_ids),
            prefill_time,
            decode_time,
            total_time,
            decode_tokens,
            throughput,
        )
        return full_text
