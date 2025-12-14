"""CUDA-only backend using Qwen3Model + sgl_kernel attention."""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Tuple, cast

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from utils.runtime import configure_torch, inference_context

from ..cache import CacheStats, PrefillCache, PrefixCache
from .sgl_kernel_backend import KVPageState, sgl_kernel_available
from .sglang_qwen import Qwen3DecoderLayer, Qwen3Model

logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str) -> str:
    override = os.getenv("MODEL_LOCAL_DIR")
    if override:
        logger.info("Using MODEL_LOCAL_DIR=%s", override)
        return override
    return model_name


class SglKernelQwenBackend:
    """Qwen3 + sgl_kernel backend (CUDA only)."""

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        if not device.startswith("cuda"):
            raise RuntimeError("SglKernelQwenBackend requires CUDA device")
        if not sgl_kernel_available():
            raise RuntimeError("sgl_kernel is not available on this system")

        configure_torch(device)
        model_path = resolve_model_path(model_name)
        trust_remote_code = os.getenv("TRUST_REMOTE_CODE", "0") != "0"
        self.device = device
        self.model_name = model_name

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )
        self.model = Qwen3Model(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            max_position_embeddings=getattr(cfg, "max_position_embeddings", 32768),
            rms_norm_eps=getattr(cfg, "rms_norm_eps", 1e-6),
        )
        self._load_weights(model_path, trust_remote_code=trust_remote_code)
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

        self.max_context_length = getattr(cfg, "max_position_embeddings", None)
        self.max_context_margin = int(os.getenv("MAX_CONTEXT_MARGIN", "16"))

        self._planned_max_new_tokens: int | None = None

        self._init_caches()
        logger.info(
            "sgl_kernel backend ready | model=%s device=%s eos_token_id=%d",
            model_name,
            device,
            self.eos_token_id,
        )

    # ---------------- Token helpers ----------------
    def tokenize(self, prompt: str) -> List[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ---------------- Generation ----------------
    def cap_max_new_tokens(self, prompt_len: int, requested: int | None) -> int | None:
        return requested

    def prepare_generation(self, max_new_tokens: int | None) -> None:
        self._planned_max_new_tokens = max_new_tokens

    def prefill_forward(self, prompt_ids: List[int], use_context: bool = True) -> Tuple[int, Any]:
        logger.info("Prefill (sgl_kernel) | tokens=%d", len(prompt_ids))
        self._reset_kv_cache()

        cache_key, cached_kv = self._maybe_get_prefill_cache(prompt_ids)
        if cached_kv is not None:
            assert cache_key is not None
            self._restore_kv_cache(cached_kv)
            return cache_key, cached_kv

        prefix_hit = self._maybe_get_prefix_cache(prompt_ids)
        if prefix_hit is not None:
            tokens, cached_kv = prefix_hit
            self._restore_kv_cache(cached_kv)
            remaining = prompt_ids[len(tokens) :]
            logits, _ = self._run_model(
                remaining, start_pos=len(tokens), use_context=use_context
            )
        else:
            logits, _ = self._run_model(prompt_ids, start_pos=0, use_context=use_context)

        first_token_id = int(torch.argmax(logits, dim=-1).item())
        kv_snapshot = self._snapshot_kv_cache()
        self._update_prefill_cache(prompt_ids, first_token_id, kv_snapshot)
        self._update_prefix_cache(prompt_ids, kv_snapshot)
        logger.info(
            "Prefill emitted token_id=%d text=%r",
            first_token_id,
            self.decode_tokens([first_token_id]),
        )
        return first_token_id, kv_snapshot

    def decode_forward(
        self, last_token_id: int, kv_cache: Any, use_context: bool = True
    ) -> Tuple[int, Any]:
        self._restore_kv_cache(kv_cache)
        current_len = self._kv_length(kv_cache)
        logits, _ = self._run_model(
            [last_token_id], start_pos=current_len, use_context=use_context
        )
        next_token_id = int(torch.argmax(logits, dim=-1).item())
        kv_snapshot = self._snapshot_kv_cache()
        return next_token_id, kv_snapshot

    # ---------------- Cache helpers ----------------
    def longest_prefix_match_length(self, prompt_ids: List[int]) -> int:
        return self.prefix_cache.match_length(prompt_ids)

    def insert_prefix(self, prompt: str) -> None:
        if not self.enable_prefix_cache or self.prefix_cache_size <= 0:
            return
        prompt_ids = self.tokenize(prompt)
        if not prompt_ids:
            return
        self.prefill_forward(prompt_ids)

    def cache_metrics(self) -> Dict[str, int]:
        return self.cache_stats.as_dict()

    # ---------------- Internal ----------------
    def _init_caches(self) -> None:
        self.prefill_cache_size = int(os.getenv("PREFILL_CACHE_SIZE", "4"))
        self.prefill_cache_token_budget = int(os.getenv("PREFILL_CACHE_TOKEN_BUDGET", "32768"))
        self.enable_prefix_cache = os.getenv("PREFIX_CACHE", "1") != "0"
        self.prefix_cache_size = int(os.getenv("PREFIX_CACHE_SIZE", "8"))
        self.prefix_cache_max_tokens = int(os.getenv("PREFIX_CACHE_MAX_TOKENS", "2048"))
        self.prefix_cache_token_budget = int(os.getenv("PREFIX_CACHE_TOKEN_BUDGET", "32768"))
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

    def _run_model(
        self, token_ids: List[int], start_pos: int, use_context: bool
    ) -> Tuple[torch.Tensor, Any]:
        input_ids = torch.as_tensor(token_ids, device=self.device).unsqueeze(0)
        auto_ctx, inf_ctx = (
            inference_context(self.device) if use_context else (nullcontext(), nullcontext())
        )
        with auto_ctx, inf_ctx:
            logits = self.model(input_ids, start_pos=start_pos)
        logits = logits[:, -1, :]
        return logits, None

    def _maybe_get_prefill_cache(
        self, prompt_ids: List[int]
    ) -> tuple[int, Any] | tuple[None, None]:
        return self.prefill_cache.maybe_get(prompt_ids, self.cache_stats)

    def _maybe_get_prefix_cache(self, prompt_ids: List[int]) -> tuple[Tuple[int, ...], Any] | None:
        return self.prefix_cache.maybe_get(prompt_ids, self.cache_stats)

    def _update_prefill_cache(
        self, prompt_ids: List[int], first_token_id: int, kv_cache: Any
    ) -> None:
        self.prefill_cache.update(prompt_ids, first_token_id, kv_cache)

    def _update_prefix_cache(self, prompt_ids: List[int], kv_cache: Any) -> None:
        self.prefix_cache.update(prompt_ids, kv_cache)

    def _iter_decoder_layers(self) -> Iterable[Qwen3DecoderLayer]:
        return cast(Iterable[Qwen3DecoderLayer], self.model.layers)

    def _reset_kv_cache(self) -> None:
        for layer in self._iter_decoder_layers():
            layer.attention.attn.page_state = None

    def _snapshot_kv_cache(self) -> List[KVPageState | None]:
        snaps: List[KVPageState | None] = []
        for layer in self._iter_decoder_layers():
            state = layer.attention.attn.page_state
            if state is None:
                snaps.append(None)
                continue
            snaps.append(
                KVPageState(
                    k_cache=state.k_cache.clone(),
                    v_cache=state.v_cache.clone(),
                    page_table=state.page_table.clone(),
                    cache_seqlens=state.cache_seqlens.clone(),
                )
            )
        return snaps

    def _restore_kv_cache(self, snapshots: List[KVPageState | None]) -> None:
        if snapshots is None:
            self._reset_kv_cache()
            return
        for layer, snap in zip(self._iter_decoder_layers(), snapshots):
            layer.attention.attn.page_state = snap

    def _kv_length(self, snapshots: List[KVPageState | None]) -> int:
        if snapshots is None or not snapshots:
            return 0
        for snap in snapshots:
            if snap is not None and snap.cache_seqlens is not None:
                return int(snap.cache_seqlens[0].item())
        return 0

    def _load_weights(self, model_path: str, trust_remote_code: bool) -> None:
        logger.info("Loading HF weights for Qwen3 from %s", model_path)
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, device_map="cpu"
        )
        sd = hf_model.state_dict()
        del hf_model

        with torch.no_grad():
            if "model.embed_tokens.weight" in sd:
                self.model.embed_tokens.weight.copy_(sd["model.embed_tokens.weight"])
            if "model.norm.weight" in sd:
                self.model.norm.weight.copy_(sd["model.norm.weight"])
            if "lm_head.weight" in sd:
                self.model.lm_head.weight.copy_(sd["lm_head.weight"])

            layers = list(self._iter_decoder_layers())
            for idx, layer in enumerate(layers):
                prefix = f"model.layers.{idx}."
                for attr, key in (
                    ("input_layernorm.weight", prefix + "input_layernorm.weight"),
                    ("post_attention_layernorm.weight", prefix + "post_attention_layernorm.weight"),
                ):
                    if key in sd:
                        getattr(layer, attr.split(".")[0]).weight.copy_(sd[key])

                q_key = prefix + "self_attn.q_proj.weight"
                k_key = prefix + "self_attn.k_proj.weight"
                v_key = prefix + "self_attn.v_proj.weight"
                if all(k in sd for k in (q_key, k_key, v_key)):
                    q_w = sd[q_key]
                    k_w = sd[k_key]
                    v_w = sd[v_key]
                    layer.attention.qkv_proj.proj.weight.copy_(torch.cat([q_w, k_w, v_w], dim=0))
                o_key = prefix + "self_attn.o_proj.weight"
                if o_key in sd:
                    layer.attention.o_proj.weight.copy_(sd[o_key])

                gate_key = prefix + "mlp.gate_proj.weight"
                up_key = prefix + "mlp.up_proj.weight"
                down_key = prefix + "mlp.down_proj.weight"
                if gate_key in sd and up_key in sd:
                    layer.mlp.gate_up.weight.copy_(torch.cat([sd[gate_key], sd[up_key]], dim=0))
                if down_key in sd:
                    layer.mlp.down.weight.copy_(sd[down_key])

        logger.info("Loaded Qwen3 weights into sgl_kernel backend")
