"""Engine layer implementing prefill + decode with streaming callbacks.

This is the mini analogue of sglang's engine: it orchestrates per-request
state, prefill + decode loops, and streaming callbacks on top of the backend.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional

from backend.model_backend import ModelBackend
from optimizations import inference_context

logger = logging.getLogger(__name__)


class SGLangMiniEngine:
    """Simple engine orchestrating generation using a ModelBackend."""

    def __init__(
        self,
        backend: ModelBackend,
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
        prompt_ids = prompt_ids if prompt_ids is not None else self.backend.tokenize(prompt)
        requested_tokens = max_new_tokens or self.max_new_tokens_default
        max_tokens = self.backend.cap_max_new_tokens(len(prompt_ids), requested_tokens)
        if max_tokens is None:
            max_tokens = requested_tokens

        logger.info(
            "Starting generation run | prompt_tokens=%d max_new_tokens=%d",
            len(prompt_ids),
            max_tokens,
        )

        generated_ids: list[int] = []
        kv_cache = None
        eos_token_id = self.backend.eos_token_id
        finished = False

        auto_ctx, inf_ctx = inference_context(self.backend.device)
        text_chunks: list[str] = []
        t_start = time.perf_counter()
        with auto_ctx, inf_ctx:
            t_prefill_start = time.perf_counter()
            first_token_id, kv_cache = self.backend.prefill_forward(
                prompt_ids, use_context=False
            )
            t_prefill_end = time.perf_counter()
            generated_ids.append(first_token_id)
            text_delta = self.backend.decode_tokens([first_token_id])
            text_chunks.append(text_delta)
            logger.info("Prefill emitted token_id=%d text=%r", first_token_id, text_delta)
            stream_callback(text_delta)

            if first_token_id == eos_token_id or max_tokens == 1:
                finished = True
            else:
                t_decode_start = time.perf_counter()
                for step_index in range(max_tokens - 1):
                    if finished:
                        break
                    if self.backend.max_context_length is not None:
                        used = len(prompt_ids) + len(generated_ids)
                        budget_left = (
                            self.backend.max_context_length - used - self.backend.max_context_margin
                        )
                        if budget_left <= 0:
                            logger.info(
                                "Stopping decode to avoid context overflow "
                                "(used=%d, max_ctx=%d, margin=%d)",
                                used,
                                self.backend.max_context_length,
                                self.backend.max_context_margin,
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
                    text_delta = self.backend.decode_tokens([next_token_id])
                    should_log = (
                        step_index == 0
                        or (step_index + 1) % max(1, self.decode_log_stride) == 0
                        or finished
                    )
                    if should_log:
                        logger.info(
                            "Decode step %d emitted token_id=%d text=%r finished=%s step_time=%.4fs",
                            step_index,
                            next_token_id,
                            text_delta,
                            finished,
                            step_end - step_start,
                        )
                    stream_callback(text_delta)
                    text_chunks.append(text_delta)
                t_decode_end = time.perf_counter()

            # We already decoded per-step deltas; join them instead of re-decoding full ids.
            full_text = "".join(text_chunks)
            finished = True
        t_end = time.perf_counter()
        prefill_time = (t_prefill_end - t_prefill_start) if "t_prefill_end" in locals() else 0.0
        decode_time = (t_decode_end - t_decode_start) if "t_decode_end" in locals() else 0.0
        total_time = t_end - t_start
        decode_tokens = max(0, len(generated_ids) - 1)
        throughput = (len(generated_ids) / total_time) if total_time > 0 else 0.0
        logger.info(
            "Generation complete | tokens=%d prefill=%.3fs decode=%.3fs total=%.3fs "
            "decode_tokens=%d throughput=%.2f tok/s",
            len(generated_ids),
            prefill_time,
            decode_time,
            total_time,
            decode_tokens,
            throughput,
        )
        return full_text
