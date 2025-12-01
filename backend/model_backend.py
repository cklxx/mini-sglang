"""Model backend that wraps the transformer model and tokenizer.

It mirrors sglang's model-backend layer in a tiny form: loading the
tokenizer/model, exposing prefill/decode with KV cache reuse, and hiding
device/KV details from the engine.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from optimizations import configure_torch, inference_context, maybe_compile_model


logger = logging.getLogger(__name__)


class ModelBackend:
    """Backend that handles model/tokenizer loading and forward passes."""

    def __init__(self, model_name: str, device: str, compile_model: bool = False) -> None:
        compile_model = compile_model or os.getenv("COMPILE_MODEL", "0") == "1"
        configure_torch(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            # Align pad with EOS so we can build explicit attention masks.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = maybe_compile_model(self.model, device=device, enabled=compile_model)
        self.model.to(device)
        self.model.eval()

        self.device = device
        # Simple LRU caches for tokenization and prefill KV reuse.
        self.token_cache_size = int(os.getenv("TOKEN_CACHE_SIZE", "32"))
        self.prefill_cache_size = int(os.getenv("PREFILL_CACHE_SIZE", "8"))
        self.token_cache: OrderedDict[str, List[int]] = OrderedDict()
        self.prefill_cache: OrderedDict[str, Tuple[int, Any]] = OrderedDict()
        self.decode_buffer_enabled = os.getenv("DECODE_BUFFER", "1") != "0"
        self._decode_buffer = torch.empty(
            (1, 1), device=self.device, dtype=torch.long
        ) if self.decode_buffer_enabled else None
        eos_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.sep_token_id
        )
        if eos_id is None:
            eos_id = self.tokenizer.pad_token_id
        if eos_id is None:
            eos_id = 0
        self.eos_token_id = eos_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id
        if self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        logger.info(
            "Model backend ready: model=%s device=%s eos_token_id=%s",
            model_name,
            device,
            self.eos_token_id,
        )

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def tokenize(self, prompt: str) -> List[int]:
        """Convert prompt text to token ids."""
        if self.token_cache_size > 0 and prompt in self.token_cache:
            token_ids = self.token_cache.pop(prompt)
            self.token_cache[prompt] = token_ids  # mark as most recently used
            logger.debug("Tokenized prompt (cache hit) into %d tokens", len(token_ids))
            return token_ids

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.token_cache_size > 0:
            self.token_cache[prompt] = token_ids
            if len(self.token_cache) > self.token_cache_size:
                self.token_cache.popitem(last=False)
        logger.debug("Tokenized prompt into %d tokens", len(token_ids))
        return token_ids

    def decode_tokens(self, token_ids: List[int]) -> str:
        """Decode token ids back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Streaming-friendly forward passes (prefill + decode)
    # ------------------------------------------------------------------

    def prefill_forward(self, prompt_ids: List[int]) -> Tuple[int, Any]:
        """Run the prefill (first) forward pass with KV cache enabled."""
        input_ids = torch.as_tensor(prompt_ids, device=self.device).unsqueeze(0)
        logger.info(
            "Running prefill_forward with sequence length=%d on device=%s",
            input_ids.shape[1],
            self.device,
        )

        prompt_key = None
        if self.prefill_cache_size > 0:
            prompt_key = ",".join(str(i) for i in prompt_ids)
            if prompt_key in self.prefill_cache:
                first_token_id, cached_kv = self.prefill_cache.pop(prompt_key)
                self.prefill_cache[prompt_key] = (first_token_id, cached_kv)
                logger.info(
                    "Prefill cache hit for seq_len=%d (first_token_id=%d)",
                    input_ids.shape[1],
                    first_token_id,
                )
                return int(first_token_id), cached_kv

        auto_ctx, inf_ctx = inference_context(self.device)
        with auto_ctx, inf_ctx:
            outputs = self.model(input_ids=input_ids, use_cache=True)
        logits = outputs.logits[:, -1, :]
        first_token_id = int(torch.argmax(logits, dim=-1).item())
        kv_cache = outputs.past_key_values

        if prompt_key is not None:
            self.prefill_cache[prompt_key] = (first_token_id, kv_cache)
            if len(self.prefill_cache) > self.prefill_cache_size:
                self.prefill_cache.popitem(last=False)
        logger.info(
            "Prefill produced first_token_id=%d (text=%r)",
            first_token_id,
            self.decode_tokens([first_token_id]),
        )
        return first_token_id, kv_cache

    def decode_forward(self, last_token_id: int, kv_cache: Any) -> Tuple[int, Any]:
        """Run a single decode step using the existing KV cache."""
        if self._decode_buffer is not None:
            self._decode_buffer[0, 0] = last_token_id
            input_ids = self._decode_buffer
        else:
            input_ids = torch.as_tensor([[last_token_id]], device=self.device)
        logger.debug(
            "Running decode_forward with last_token_id=%d (text=%r)",
            last_token_id,
            self.decode_tokens([last_token_id]),
        )
        auto_ctx, inf_ctx = inference_context(self.device)
        with auto_ctx, inf_ctx:
            outputs = self.model(
                input_ids=input_ids, past_key_values=kv_cache, use_cache=True
            )
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).item()
        new_kv_cache = outputs.past_key_values
        logger.debug(
            "Decode step produced next_token_id=%d (text=%r)",
            next_token_id,
            self.decode_tokens([int(next_token_id)]),
        )
        return next_token_id, new_kv_cache

    # ------------------------------------------------------------------
    # Batched/non-streaming generation
    # ------------------------------------------------------------------
    def generate_greedy(self, prompt_ids: List[int], max_new_tokens: int) -> List[int]:
        """Run a full greedy generation in a single call for benchmarking.

        Returns only the newly generated token ids (excluding the prompt).
        """

        input_ids = torch.as_tensor(prompt_ids, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        logger.info(
            "Running generate_greedy with seq_len=%d max_new_tokens=%d on %s",
            input_ids.shape[1],
            max_new_tokens,
            self.device,
        )
        auto_ctx, inf_ctx = inference_context(self.device)
        with auto_ctx, inf_ctx:
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.eos_token_id,
                top_p=None,
                top_k=None,
                temperature=None,
            )
        full_ids: List[int] = [int(x) for x in outputs[0].tolist()]
        generated_only = full_ids[len(prompt_ids) :]
        logger.info(
            "generate_greedy produced %d tokens (text preview=%r)",
            len(generated_only),
            self.decode_tokens(generated_only[:5]),
        )
        return generated_only

    def generate_streaming_baseline(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        log_stride: int = 32,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> Tuple[str, float]:
        """Stream tokens using HF's TextIteratorStreamer as a non-sglang baseline."""

        input_ids = torch.as_tensor(prompt_ids, device=self.device).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        log_stride = max(1, int(os.getenv("BASELINE_STREAM_LOG_STRIDE", str(log_stride))))

        def _generate() -> None:
            auto_ctx, inf_ctx = inference_context(self.device)
            with auto_ctx, inf_ctx:
                self.model.generate(
                    input_ids=input_ids,
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
                    "[hf-stream] chunk %03d: %r (log_stride=%d)", idx, token_text, log_stride
                )

        thread.join()
        duration = time.perf_counter() - start
        full_text = "".join(chunks)
        logger.info(
            "HF streaming baseline complete | chunks=%d duration=%.3fs tokens=%d",
            len(chunks),
            duration,
            len(self.tokenize(full_text)),
        )
        return full_text, duration
