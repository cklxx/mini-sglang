"""Engine layer implementing prefill + decode with streaming callbacks.

This is the mini analogue of sglang's engine: it orchestrates per-request
state, prefill + decode loops, and streaming callbacks on top of the backend.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from backend.model_backend import ModelBackend


logger = logging.getLogger(__name__)


@dataclass
class RequestState:
    prompt_text: str
    prompt_ids: List[int]
    generated_ids: List[int]
    max_new_tokens: int
    eos_token_id: int
    kv_cache: Any | None
    finished: bool


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
        self.decode_log_stride = decode_log_stride or int(
            os.getenv("DECODE_LOG_STRIDE", "32")
        )

    def run_generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int],
        stream_callback: Callable[[str], None],
    ) -> str:
        """Generate text using prefill + decode while invoking a stream callback."""
        prompt_ids = self.backend.tokenize(prompt)
        requested_tokens = max_new_tokens or self.max_new_tokens_default
        max_tokens = self.backend.cap_max_new_tokens(len(prompt_ids), requested_tokens)
        if max_tokens is None:
            max_tokens = requested_tokens

        logger.info(
            "Starting generation run | prompt_tokens=%d max_new_tokens=%d",
            len(prompt_ids),
            max_tokens,
        )

        state = RequestState(
            prompt_text=prompt,
            prompt_ids=prompt_ids,
            generated_ids=[],
            max_new_tokens=max_tokens,
            eos_token_id=self.backend.eos_token_id,
            kv_cache=None,
            finished=False,
        )

        first_token_id, kv_cache = self.backend.prefill_forward(prompt_ids)
        state.generated_ids.append(first_token_id)
        state.kv_cache = kv_cache
        text_delta = self.backend.decode_tokens([first_token_id])
        text_chunks: list[str] = [text_delta]
        logger.info("Prefill emitted token_id=%d text=%r", first_token_id, text_delta)
        stream_callback(text_delta)

        if first_token_id == state.eos_token_id or state.max_new_tokens == 1:
            state.finished = True
        else:
            for step_index in range(state.max_new_tokens - 1):
                if state.finished:
                    break
                if self.backend.max_context_length is not None:
                    used = len(state.prompt_ids) + len(state.generated_ids)
                    budget_left = self.backend.max_context_length - used - self.backend.max_context_margin
                    if budget_left <= 0:
                        logger.info(
                            "Stopping decode to avoid context overflow "
                            "(used=%d, max_ctx=%d, margin=%d)",
                            used,
                            self.backend.max_context_length,
                            self.backend.max_context_margin,
                        )
                        state.finished = True
                        break
                last_token_id = state.generated_ids[-1]
                next_token_id, kv_cache = self.backend.decode_forward(
                    last_token_id, state.kv_cache
                )
                state.generated_ids.append(next_token_id)
                state.kv_cache = kv_cache
                if next_token_id == state.eos_token_id:
                    state.finished = True
                text_delta = self.backend.decode_tokens([next_token_id])
                should_log = (
                    step_index == 0
                    or (step_index + 1) % max(1, self.decode_log_stride) == 0
                    or state.finished
                )
                if should_log:
                    logger.info(
                        "Decode step %d emitted token_id=%d text=%r finished=%s",
                        step_index,
                        next_token_id,
                        text_delta,
                        state.finished,
                    )
                stream_callback(text_delta)
                text_chunks.append(text_delta)

        # We already decoded per-step deltas; join them instead of re-decoding full ids.
        full_text = "".join(text_chunks)
        logger.info("Generation complete | %d tokens emitted", len(state.generated_ids))
        return full_text
