"""Simplified torch/HF backend for CPU/MPS with prefix/prefill cache."""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from contextlib import nullcontext
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.cache import CacheStats, PrefillCache, PrefixCache
from utils.runtime import configure_torch, inference_context

logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str) -> str:
    override = os.getenv("MODEL_LOCAL_DIR")
    if override:
        logger.info("Using MODEL_LOCAL_DIR=%s", override)
        return override
    return model_name


class ModelBackend:
    """Minimal torch backend for CPU/MPS."""

    def __init__(self, model_name: str, device: str) -> None:
        configure_torch(device)
        model_path = resolve_model_path(model_name)
        trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "0") != "0"

        self.device = device
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model.to(self.device)
        self.model.eval()

        eos_id = (
            self.tokenizer.eos_token_id
            if self.tokenizer.eos_token_id is not None
            else self.tokenizer.pad_token_id
        )
        self.eos_token_id = int(eos_id if eos_id is not None else 0)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.eos_token_id

        self.max_context_length = getattr(self.model.config, "max_position_embeddings", None)
        self.max_context_margin = int(os.getenv("MAX_CONTEXT_MARGIN", "16"))

        self._init_caches()
        logger.info(
            "Torch backend ready | model=%s device=%s eos_token_id=%d",
            model_name,
            device,
            self.eos_token_id,
        )

    # ---------------- Token helpers ----------------
    def tokenize(self, prompt: str) -> List[int]:
        if self.token_cache_size > 0 and prompt in self.token_cache:
            ids = self.token_cache.pop(prompt)
            self.token_cache[prompt] = ids
            return ids
        ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if self.token_cache_size > 0:
            self.token_cache[prompt] = ids
            if len(self.token_cache) > self.token_cache_size:
                self.token_cache.popitem(last=False)
        return ids

    def decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ---------------- Generation ----------------
    def cap_max_new_tokens(self, prompt_len: int, requested: int | None) -> int | None:
        return requested

    def prepare_generation(self, max_new_tokens: int | None) -> None:
        self._planned_max_new_tokens = max_new_tokens

    def prefill_forward(self, prompt_ids: List[int], use_context: bool = True) -> Tuple[int, Any]:
        seq_len = len(prompt_ids)
        logger.info("Prefill | seq_len=%d device=%s", seq_len, self.device)

        cache_key, cached_kv = self._maybe_get_prefill_cache(prompt_ids)
        if cached_kv is not None:
            return cache_key, cached_kv

        prefix_hit = self._maybe_get_prefix_cache(prompt_ids)
        if prefix_hit is not None:
            prefix_tokens, cached_kv = prefix_hit
            remaining = prompt_ids[len(prefix_tokens) :]
            logits, kv_cache = self._run_model(
                remaining, past_kv=cached_kv, use_context=use_context
            )
        else:
            logits, kv_cache = self._run_model(prompt_ids, past_kv=None, use_context=use_context)

        first_token_id = int(torch.argmax(logits, dim=-1).item())
        self._update_prefill_cache(prompt_ids, first_token_id, kv_cache)
        self._update_prefix_cache(prompt_ids, kv_cache)
        logger.info(
            "Prefill emitted token_id=%d text=%r",
            first_token_id,
            self.decode_tokens([first_token_id]),
        )
        return first_token_id, kv_cache

    def decode_forward(
        self, last_token_id: int, kv_cache: Any, use_context: bool = True
    ) -> Tuple[int, Any]:
        input_ids = torch.as_tensor([[last_token_id]], device=self.device)
        logits, new_kv = self._run_model(input_ids, past_kv=kv_cache, use_context=use_context)
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        return next_token_id, new_kv

    # ---------------- Cache helpers ----------------
    def longest_prefix_match_length(self, prompt_ids: List[int]) -> int:
        return self.prefix_cache.match_length(prompt_ids)

    def insert_prefix(self, prompt: str) -> None:
        if not self.enable_prefix_cache or self.prefix_cache_size <= 0:
            return
        prompt_ids = self.tokenize(prompt)
        if not prompt_ids:
            return
        _, kv = self.prefill_forward(prompt_ids)
        self._update_prefix_cache(prompt_ids, kv)

    def cache_metrics(self) -> Dict[str, int]:
        return self.cache_stats.as_dict()

    # ---------------- Internal ----------------
    def _init_caches(self) -> None:
        self.token_cache_size = int(os.getenv("TOKEN_CACHE_SIZE", "32"))
        self.prefill_cache_size = int(os.getenv("PREFILL_CACHE_SIZE", "4"))
        self.prefill_cache_token_budget = int(os.getenv("PREFILL_CACHE_TOKEN_BUDGET", "32768"))
        self.enable_prefix_cache = os.getenv("PREFIX_CACHE", "1") != "0"
        self.prefix_cache_size = int(os.getenv("PREFIX_CACHE_SIZE", "8"))
        self.prefix_cache_max_tokens = int(os.getenv("PREFIX_CACHE_MAX_TOKENS", "2048"))
        self.prefix_cache_token_budget = int(os.getenv("PREFIX_CACHE_TOKEN_BUDGET", "32768"))

        self.token_cache: OrderedDict[str, List[int]] = OrderedDict()
        self.cache_stats = CacheStats()
        self.prefill_cache = PrefillCache(
            size=self.prefill_cache_size, token_budget=self.prefill_cache_token_budget
        )
        self.prefix_cache = PrefixCache(
            enable=self.enable_prefix_cache,
            size=self.prefix_cache_size,
            max_tokens=self.prefix_cache_max_tokens,
            token_budget=self.prefix_cache_token_budget,
            policy="lru",
        )
        self._planned_max_new_tokens: int | None = None

    def _run_model(
        self,
        token_ids: List[int] | torch.Tensor,
        past_kv: Any,
        use_context: bool,
    ) -> Tuple[torch.Tensor, Any]:
        if isinstance(token_ids, list):
            input_ids = torch.as_tensor(token_ids, device=self.device).unsqueeze(0)
        else:
            input_ids = token_ids
        auto_ctx, inf_ctx = (
            inference_context(self.device) if use_context else (nullcontext(), nullcontext())
        )
        with auto_ctx, inf_ctx:
            outputs = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
        logits = outputs.logits[:, -1, :]
        return logits, outputs.past_key_values

    def _maybe_get_prefill_cache(self, prompt_ids: List[int]) -> Tuple[int | None, Any | None]:
        return self.prefill_cache.maybe_get(prompt_ids, self.cache_stats)

    def _maybe_get_prefix_cache(self, prompt_ids: List[int]) -> tuple[Tuple[int, ...], Any] | None:
        return self.prefix_cache.maybe_get(prompt_ids, self.cache_stats)

    def _update_prefill_cache(
        self, prompt_ids: List[int], first_token_id: int, kv_cache: Any
    ) -> None:
        self.prefill_cache.update(prompt_ids, first_token_id, kv_cache)

    def _update_prefix_cache(self, prompt_ids: List[int], kv_cache: Any) -> None:
        self.prefix_cache.update(prompt_ids, kv_cache)
