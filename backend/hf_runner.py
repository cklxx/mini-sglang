"""Lightweight HF-only generation path decoupled from mini-sglang backend."""

from __future__ import annotations

import logging
import os
import time
import threading
from typing import Callable, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

logger = logging.getLogger(__name__)


def _resolve_model_path(model_name: str) -> str:
    """Local resolver to avoid coupling with mini-sglang backend."""
    local_override = os.getenv("MODEL_LOCAL_DIR")
    if local_override:
        logger.info("HFBaseline: Using MODEL_LOCAL_DIR=%s", local_override)
        return local_override
    return model_name


class HFBaseline:
    """Standalone HF generation helper (no mini-sglang caches or backends)."""

    def __init__(self, model_name: str, device: str) -> None:
        model_path = _resolve_model_path(model_name)
        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self._lock = threading.Lock()
        self.eos_token_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.pad_token_id
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id or 0

    def tokenize(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: int,
        log_stride: int = 32,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, float]:
        """Stream tokens using HF's TextIteratorStreamer (no mini-sglang caches)."""

        prompt_ids = torch.as_tensor(self.tokenize(prompt), device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(prompt_ids)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        log_stride = max(1, int(os.getenv("BASELINE_STREAM_LOG_STRIDE", str(log_stride))))

        def _generate() -> None:
            self.model.generate(
                input_ids=prompt_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id,
                streamer=streamer,
                top_p=None,
                top_k=None,
                temperature=None,
            )

        import threading

        with self._lock:
            start = time.perf_counter()

            thread = threading.Thread(target=_generate)
            thread.start()

            chunks: list[str] = []
            for idx, token_text in enumerate(streamer, 1):
                if stream_callback:
                    stream_callback(token_text)
                chunks.append(token_text)
                if idx == 1 or idx % log_stride == 0:
                    logger.info(
                        "[hf-baseline] chunk %03d: %r (log_stride=%d)", idx, token_text, log_stride
                    )

            thread.join()
            duration = time.perf_counter() - start

        full_text = "".join(chunks)
        logger.info(
            "HF baseline complete | chunks=%d tokens=%d", len(chunks), len(self.tokenize(full_text))
        )
        return full_text, duration
