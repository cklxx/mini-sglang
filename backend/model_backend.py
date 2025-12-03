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
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.cache_utils import StaticCache

try:
    from transformers.utils import is_flash_attn_2_available
except Exception:

    def is_flash_attn_2_available() -> bool:
        return False

from backend.cache import CacheStats, KVPageManager, PrefillCache, PrefixCache
from backend.attention_patch import patch_model_with_sgl_kernel
from backend.sgl_kernel_backend import KVPageState, SglKernelAttentionBackend, sgl_kernel_available
from optimizations import configure_torch, inference_context, maybe_compile_model

logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str) -> str:
    """Resolve the model path or repo id, honoring a local override."""

    local_override = os.getenv("MODEL_LOCAL_DIR")
    if local_override:
        logger.info("Using MODEL_LOCAL_DIR=%s", local_override)
        return local_override

    return model_name


class ModelBackend:
    """Backend that handles model/tokenizer loading and forward passes."""

    def __init__(self, model_name: str, device: str, compile_model: bool = False) -> None:
        configure_torch(device)
        model_path = resolve_model_path(model_name)

        self.device = device
        self.model_name = model_name
        compile_enabled = self._flag_from_env("COMPILE_MODEL", default=compile_model)
        self.enable_static_kv = self._flag_from_env(
            "ENABLE_STATIC_KV", default=self.device.startswith("cuda")
        )
        self.static_kv_max_len = int(os.getenv("STATIC_KV_MAX_LEN", "0"))
        self._static_kv_budget = int(os.getenv("STATIC_KV_DEFAULT_BUDGET", "2048"))
        self.enable_cuda_graph = (
            self.device.startswith("cuda") and os.getenv("ENABLE_CUDA_GRAPH", "1") != "0"
        )
        self.prefill_graph_seq_len = int(os.getenv("PREFILL_GRAPH_SEQ_LEN", "0"))
        self.prefill_graph_max_len = int(os.getenv("PREFILL_GRAPH_MAX_LEN", "2048"))
        self._prefill_graph_dynamic_len = self.prefill_graph_seq_len <= 0
        self.tensor_parallel_size = self._resolve_tensor_parallel_size()
        use_tensor_parallel = self._can_use_tensor_parallel()
        model_kwargs = self._tensor_parallel_kwargs(use_tensor_parallel)
        torch_dtype = self._resolve_torch_dtype()
        self.attn_impl = self._resolve_attn_impl()
        sgl_available = sgl_kernel_available() and self.device.startswith("cuda")
        sgl_default = os.getenv("ATTN_BACKEND_SGL_KERNEL") is None and sgl_available
        self.use_sgl_kernel_requested = self._flag_from_env(
            "ATTN_BACKEND_SGL_KERNEL", default=sgl_default
        )
        if self.use_sgl_kernel_requested and self.attn_impl == "flash_attention_2":
            logger.info("Using sgl_kernel backend; forcing attn_implementation=eager")
            self.attn_impl = "eager"

        if (
            torch_dtype is None
            and self.attn_impl == "flash_attention_2"
            and self.device.startswith("cuda")
        ):
            torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info(
                "Defaulting torch_dtype=%s for flash attention on CUDA", torch_dtype
            )
        if torch_dtype == torch.float32 and self.attn_impl == "flash_attention_2":
            logger.warning(
                "Flash attention requires float16/bfloat16; falling back to sdpa for torch_dtype=float32"
            )
            self.attn_impl = "sdpa"
        if self.attn_impl == "flash_attention_2" and self.enable_static_kv:
            logger.info("Disabling StaticCache when using flash attention")
            self.enable_static_kv = False

        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if self.attn_impl is not None:
            model_kwargs["attn_implementation"] = self.attn_impl
        trust_remote_code = self._flag_from_env("TRUST_REMOTE_CODE", default=False)

        self.tokenizer = self._load_tokenizer(model_path, trust_remote_code=trust_remote_code)
        self.model = self._load_model(
            model_path=model_path,
            model_kwargs=model_kwargs,
            trust_remote_code=trust_remote_code,
        )
        self._maybe_optimize_qwen_attention()
        self.model = maybe_compile_model(self.model, device=device, enabled=compile_enabled)

        if use_tensor_parallel:
            # Use a canonical device for tensor placement when the model is sharded.
            self.device = "cuda:0"
        else:
            self.model.to(self.device)
        self.model.eval()

        self._init_caches()
        self._init_chunked_prefill()
        self._init_decode_buffer()
        self._static_cache: StaticCache | None = None
        self._init_cuda_graph_state()
        self._detect_dynamic_cache()
        self._init_generation_config()
        self._init_sgl_kernel_backend()

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

    def _prefill_call(
        self, input_ids: torch.Tensor, past_key_values: Any | None, use_context: bool = True
    ):
        if use_context:
            auto_ctx, inf_ctx = inference_context(self.device)
        else:
            auto_ctx, inf_ctx = nullcontext(), nullcontext()
        with auto_ctx, inf_ctx:
            return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

    def _decode_call(
        self, input_ids: torch.Tensor, past_key_values: Any, use_context: bool = True
    ) -> Any:
        """Decode step (single-token decode)."""
        if use_context:
            auto_ctx, inf_ctx = inference_context(self.device)
        else:
            auto_ctx, inf_ctx = nullcontext(), nullcontext()
        with auto_ctx, inf_ctx:
            return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)

    def prefill_forward(self, prompt_ids: List[int], use_context: bool = True) -> Tuple[int, Any]:
        """Run the prefill (first) forward pass with KV cache enabled."""
        seq_len = len(prompt_ids)
        logger.info(
            "Running prefill_forward with sequence length=%d on device=%s",
            seq_len,
            self.device,
        )
        cache_key, cached_kv = self._maybe_get_prefill_cache(prompt_ids)
        if cached_kv is not None:
            assert cache_key is not None
            return cache_key, cached_kv

        prefix_hit = self._maybe_get_prefix_cache(prompt_ids)
        if prefix_hit is not None:
            tokens, cached_kv = prefix_hit
            remaining_ids = prompt_ids[len(tokens) :]
            logger.info(
                "Prefix cache hit for seq_len=%d using prefix_len=%d", len(prompt_ids), len(tokens)
            )
            logits, kv_cache = self._prefill_run(
                remaining_ids, past_key_values=cached_kv, use_context=use_context
            )
        else:
            initial_cache = self._maybe_init_static_cache(len(prompt_ids))
            logits, kv_cache = self._prefill_run(
                prompt_ids, past_key_values=initial_cache, use_context=use_context
            )
        if isinstance(kv_cache, StaticCache):
            self._static_cache = kv_cache
        first_token_id = int(torch.argmax(logits, dim=-1).item())

        self._update_prefill_cache(prompt_ids, first_token_id, kv_cache)
        self._update_prefix_cache(prompt_ids, kv_cache)
        logger.info(
            "Prefill produced first_token_id=%d (text=%r)",
            first_token_id,
            self.decode_tokens([first_token_id]),
        )
        return first_token_id, kv_cache

    def decode_forward(
        self, last_token_id: int, kv_cache: Any, use_context: bool = True
    ) -> Tuple[int, Any]:
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
        outputs = self._decode_call(input_ids, kv_cache, use_context=use_context)
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).item()
        new_kv_cache = outputs.past_key_values
        if isinstance(new_kv_cache, StaticCache):
            self._static_cache = new_kv_cache
        next_token_id = int(next_token_id)
        logger.debug(
            "Decode step produced next_token_id=%d (text=%r)",
            next_token_id,
            self.decode_tokens([next_token_id]),
        )
        return next_token_id, new_kv_cache

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flag_from_env(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value != "0"

    def _resolve_tensor_parallel_size(self) -> int:
        env_value = os.getenv("TENSOR_PARALLEL_SIZE")
        if env_value is not None:
            return int(env_value)
        if torch.cuda.is_available():
            return max(1, torch.cuda.device_count())
        return 1

    def _can_use_tensor_parallel(self) -> bool:
        return (
            self.tensor_parallel_size > 1
            and torch.cuda.is_available()
            and torch.cuda.device_count() >= self.tensor_parallel_size
        )

    def _tensor_parallel_kwargs(self, use_tensor_parallel: bool) -> Dict[str, Any]:
        if use_tensor_parallel:
            logger.info(
                "Loading model with tensor parallelism across %d GPUs", self.tensor_parallel_size
            )
            return {"device_map": "auto"}
        if self.tensor_parallel_size > 1:
            logger.warning(
                "Requested tensor parallelism (TENSOR_PARALLEL_SIZE=%d) but found only "
                "%d CUDA devices",
                self.tensor_parallel_size,
                torch.cuda.device_count(),
            )
        return {}

    def _resolve_torch_dtype(self) -> Optional[torch.dtype]:
        """Optional dtype override for model weights."""
        dtype_str = os.getenv("MODEL_DTYPE") or os.getenv("TORCH_DTYPE")
        if dtype_str is None:
            # Default to float16 on GPU/MPS to avoid autocast/dtype mismatches.
            if self.device.startswith("cuda") or self.device == "mps":
                logger.info("Defaulting torch_dtype=float16 for device=%s", self.device)
                return torch.float16
            return None
        key = dtype_str.lower()
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "auto": None,
        }
        if key not in mapping:
            logger.warning("Unrecognized MODEL_DTYPE=%s; ignoring", dtype_str)
            return None
        dtype = mapping[key]
        if dtype is None:
            # Auto-pick: favor float16 on CUDA/MPS, otherwise default.
            if self.device.startswith("cuda") or self.device == "mps":
                dtype = torch.float16
            else:
                dtype = None
        if dtype is not None:
            logger.info("Using torch_dtype=%s for model weights", dtype)
        return dtype

    def _detect_dynamic_cache(self) -> None:
        """Force legacy/static cache to avoid HF DynamicCache paths."""
        self.uses_dynamic_cache = False
        kv_cfg = getattr(self.model.config, "kv_cache_config", None)
        use_cache = getattr(self.model.config, "use_cache", True)
        coerced = False
        if isinstance(use_cache, str) and use_cache.lower() == "dynamic":
            self.model.config.use_cache = True
            coerced = True
        impl = getattr(kv_cfg, "implementation", None)
        if isinstance(impl, str) and impl.lower() == "dynamic":
            try:
                kv_cfg.implementation = "pytorch"
            except Exception:
                self.model.config.kv_cache_config = None
            coerced = True
        if coerced:
            logger.info("Coerced DynamicCache config to legacy static cache (use_cache=True)")

    def _resolve_attn_impl(self) -> Optional[str]:
        """Optional attention implementation override."""
        attn_impl = os.getenv("ATTN_IMPL") or os.getenv("ATTN_IMPLEMENTATION")
        if attn_impl is None:
            if self.device.startswith("cuda") and os.getenv("ENABLE_FLASH_ATTENTION", "1") != "0":
                attn_impl = "flash_attention_2"
            else:
                return None
        attn_impl = attn_impl.lower()
        valid = {"flash_attention_2", "sdpa", "eager", "fa3"}
        if attn_impl not in valid:
            logger.warning(
                "Unrecognized ATTN_IMPL=%s; expected one of %s",
                attn_impl,
                ", ".join(sorted(valid)),
            )
            return None
        if attn_impl == "fa3":
            # Transformers uses flash_attention_2 keyword; keep fa3 as alias.
            attn_impl = "flash_attention_2"
        if attn_impl == "flash_attention_2" and not is_flash_attn_2_available():
            logger.warning(
                "Flash attention requested but flash_attn is not installed; falling back to sdpa"
            )
            return "sdpa"
        logger.info("Using attn_implementation=%s", attn_impl)
        return attn_impl

    def _is_qwen3(self) -> bool:
        """Detect Qwen3 models by name or config."""
        name = self.model_name.lower()
        cfg = getattr(self, "model", None)
        cfg_type = ""
        if cfg is not None:
            cfg_type = str(
                getattr(cfg, "config", None) and getattr(cfg.config, "model_type", "")
            ).lower()
        return "qwen3" in name or cfg_type == "qwen3"

    def _maybe_optimize_qwen_attention(self) -> None:
        """Tune Qwen3 attention flags to match chosen implementation."""
        if self.model is None or not self._is_qwen3():
            return
        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return
        if self.attn_impl != "flash_attention_2":
            disabled = False
            if hasattr(cfg, "use_flash_attn") and cfg.use_flash_attn:
                cfg.use_flash_attn = False
                disabled = True
            if hasattr(cfg, "flash_attn") and getattr(cfg, "flash_attn"):
                cfg.flash_attn = False
                disabled = True
            if disabled:
                logger.info("Disabled Qwen3 flash attention flags (attn_impl=%s)", self.attn_impl)
            return
        if not self.device.startswith("cuda"):
            logger.info(
                "Qwen3 model detected but flash attention requires CUDA; device=%s",
                self.device,
            )
            return
        toggled = False
        if hasattr(cfg, "use_flash_attn") and cfg.use_flash_attn is False:
            cfg.use_flash_attn = True
            toggled = True
        if hasattr(cfg, "flash_attn") and getattr(cfg, "flash_attn") is False:
            cfg.flash_attn = True
            toggled = True
        if toggled:
            logger.info("Enabled Qwen3 flash attention flags for CUDA execution")

    def _cache_dtype(self) -> torch.dtype:
        """Best-effort dtype used for KV caches (match model weights)."""
        try:
            return next(self.model.parameters()).dtype
        except StopIteration:
            if self.device.startswith("cuda") or self.device == "mps":
                return torch.float16
            return torch.float32

    def _cache_values_dtype(self, cache: StaticCache | None) -> torch.dtype | None:
        """Inspect a cache object to infer the stored dtype."""
        if cache is None:
            return None
        values = getattr(cache, "values", None)
        return self._first_tensor_dtype(values)

    def _first_tensor_dtype(self, obj: Any) -> torch.dtype | None:
        """Return dtype of the first tensor inside nested lists/tuples."""
        if obj is None:
            return None
        if torch.is_tensor(obj):
            return obj.dtype
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            return self._first_tensor_dtype(obj[0])
        return None

    def _coerce_static_cache_dtype(
        self, cache: StaticCache | None, dtype: torch.dtype
    ) -> StaticCache | None:
        """Ensure StaticCache k/v tensors match the desired dtype."""
        if cache is None:
            return None
        try:
            current = self._cache_values_dtype(cache)
            if current is None:
                return cache
            if current == dtype:
                return cache
            if hasattr(cache, "values") and torch.is_tensor(cache.values):
                cache.values = cache.values.to(dtype=dtype)
            if hasattr(cache, "keys") and torch.is_tensor(cache.keys):
                cache.keys = cache.keys.to(dtype=dtype)
            current = self._cache_values_dtype(cache)
            if current == dtype:
                logger.info("Coerced StaticCache dtype to %s", dtype)
                return cache
            logger.warning(
                "Failed to coerce StaticCache dtype (cache=%s expected=%s); disabling static KV",
                current,
                dtype,
            )
            return None
        except Exception as exc:
            logger.warning("Failed to coerce StaticCache dtype (%s); disabling static KV", exc)
            return None

    def _load_tokenizer(self, model_path: str, trust_remote_code: bool = False):
        """Load tokenizer with a fallback to trust_remote_code for newer models."""
        base_kwargs: Dict[str, Any] = {}
        if trust_remote_code:
            base_kwargs["trust_remote_code"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path, **base_kwargs)
        except ValueError as exc:
            if trust_remote_code:
                raise
            logger.warning(
                "Retrying tokenizer load with trust_remote_code=True after failure: %s", exc
            )
            retry_kwargs = dict(base_kwargs)
            retry_kwargs["trust_remote_code"] = True
            tokenizer = AutoTokenizer.from_pretrained(model_path, **retry_kwargs)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self, model_path: str, model_kwargs: Dict[str, Any], trust_remote_code: bool):
        """Load model with best-effort retry enabling trust_remote_code when needed."""
        def _attempt_load(kwargs: Dict[str, Any]):
            try:
                return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            except ImportError as exc:
                if kwargs.get("attn_implementation") == "flash_attention_2":
                    logger.warning(
                        "Flash attention requested but not available (%s); retrying with sdpa",
                        exc,
                    )
                    retry_kwargs = dict(kwargs)
                    retry_kwargs["attn_implementation"] = "sdpa"
                    return AutoModelForCausalLM.from_pretrained(model_path, **retry_kwargs)
                raise

        base_kwargs = dict(model_kwargs)
        if trust_remote_code:
            base_kwargs["trust_remote_code"] = True
        try:
            return _attempt_load(base_kwargs)
        except ValueError as exc:
            if trust_remote_code:
                raise
            logger.warning(
                "Retrying model load with trust_remote_code=True after failure: %s", exc
            )
            retry_kwargs = dict(model_kwargs)
            retry_kwargs["trust_remote_code"] = True
            return _attempt_load(retry_kwargs)

    def _init_caches(self) -> None:
        # Simple LRU caches for tokenization and prefill KV reuse.
        self.token_cache_size = int(os.getenv("TOKEN_CACHE_SIZE", "32"))
        self.prefill_cache_size = int(os.getenv("PREFILL_CACHE_SIZE", "8"))
        self.prefill_cache_token_budget = int(os.getenv("PREFILL_CACHE_TOKEN_BUDGET", "65536"))
        self.enable_prefix_cache = self._flag_from_env("PREFIX_CACHE", default=True)
        self.prefix_cache_size = int(os.getenv("PREFIX_CACHE_SIZE", "16"))
        self.prefix_cache_max_tokens = int(os.getenv("PREFIX_CACHE_MAX_TOKENS", "4096"))
        self.prefix_cache_token_budget = int(os.getenv("PREFIX_CACHE_TOKEN_BUDGET", "65536"))
        self.prefix_cache_policy = (os.getenv("PREFIX_CACHE_POLICY", "lru") or "lru").lower()
        self.page_token_budget = int(os.getenv("PAGE_TOKEN_BUDGET", "0"))
        self.page_size_tokens = int(os.getenv("PAGE_SIZE_TOKENS", "512"))

        if self.attn_impl == "flash_attention_2" or self.use_sgl_kernel_requested:
            logger.info("Disabling prefill/prefix caches for flash/sgl_kernel attention")
            self.prefill_cache_size = 0
            self.prefill_cache_token_budget = 0
            self.enable_prefix_cache = False
            self.prefix_cache_size = 0
            self.prefix_cache_max_tokens = 0
            self.prefix_cache_token_budget = 0
            self.page_token_budget = 0

        self.token_cache: OrderedDict[str, List[int]] = OrderedDict()
        self.cache_stats = CacheStats()
        self.prefill_cache = PrefillCache(
            size=self.prefill_cache_size, token_budget=self.prefill_cache_token_budget
        )
        self.page_manager: KVPageManager | None = None
        if self.page_token_budget > 0:
            self.page_manager = KVPageManager(
                token_budget=self.page_token_budget, page_size=self.page_size_tokens
            )
        self.prefix_cache = PrefixCache(
            enable=self.enable_prefix_cache,
            size=self.prefix_cache_size,
            max_tokens=self.prefix_cache_max_tokens,
            token_budget=self.prefix_cache_token_budget,
            policy=self.prefix_cache_policy,
        )
        self.prefix_cache.bind_page_manager(self.page_manager)

    def _init_chunked_prefill(self) -> None:
        # Optional chunked prefill to trade small extra passes for lower peak memory.
        self.chunked_prefill_enabled = self._flag_from_env("CHUNKED_PREFILL", default=False)
        self.prefill_chunk_size = max(1, int(os.getenv("PREFILL_CHUNK_SIZE", "512")))

    def _init_decode_buffer(self) -> None:
        self.decode_buffer_enabled = self._flag_from_env("DECODE_BUFFER", default=True)
        self._decode_buffer = (
            torch.empty((1, 1), device=self.device, dtype=torch.long)
            if self.decode_buffer_enabled
            else None
        )

    def _init_cuda_graph_state(self) -> None:
        self._prefill_graph = None
        self._prefill_graph_input: torch.Tensor | None = None
        self._prefill_graph_logits: torch.Tensor | None = None
        self._prefill_graph_kv: Any | None = None
        if self.enable_cuda_graph and self.prefill_graph_seq_len > 0:
            logger.info(
                "CUDA graph prefill enabled for seq_len=%d", self.prefill_graph_seq_len
            )
        elif self.enable_cuda_graph:
            logger.info(
                "CUDA graph requested; will capture first-seen prompt len "
                "if <= PREFILL_GRAPH_MAX_LEN=%d",
                self.prefill_graph_max_len,
            )

    def _init_generation_config(self) -> None:
        self.aggressive_max_new_tokens = self._flag_from_env("AGGRESSIVE_MAX_NEW_TOKENS", True)
        self.max_context_margin = int(os.getenv("MAX_CONTEXT_MARGIN", "16"))

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
        gen_config = getattr(self.model, "generation_config", None)
        if gen_config is not None and gen_config.pad_token_id is None:
            gen_config.pad_token_id = self.tokenizer.pad_token_id
        self.max_context_length = getattr(self.model.config, "max_position_embeddings", None)

    # ------------------------------------------------------------------
    # sgl_kernel scaffolding (backend choice + cache state)
    # ------------------------------------------------------------------

    def _init_sgl_kernel_backend(self) -> None:
        use_sgl = self.use_sgl_kernel_requested
        self._sgl_kernel_backend: SglKernelAttentionBackend | None = None
        self._sgl_kernel_page_state: KVPageState | None = None
        if not use_sgl:
            logger.info("ATTN_BACKEND_SGL_KERNEL=0; using model-native attention")
            return
        if not sgl_kernel_available():
            logger.warning(
                "ATTN_BACKEND_SGL_KERNEL=1 but sgl_kernel unavailable or CUDA not found; falling back"
            )
            return
        if not hasattr(self.model.config, "num_attention_heads"):
            logger.warning("Model config missing num_attention_heads; cannot init sgl_kernel backend")
            return
        num_heads = int(self.model.config.num_attention_heads)
        num_kv_heads = int(getattr(self.model.config, "num_key_value_heads", num_heads))
        head_dim = int(self.model.config.hidden_size // num_heads)
        self._sgl_kernel_backend = SglKernelAttentionBackend(
            num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads
        )
        logger.info(
            "sgl_kernel attention backend initialized | heads=%d kv_heads=%d head_dim=%d",
            num_heads,
            num_kv_heads,
            head_dim,
        )
        wrapped = patch_model_with_sgl_kernel(self.model, self._sgl_kernel_backend)
        logger.info("Patched %d attention modules to route through sgl_kernel backend", wrapped)

    def use_sgl_kernel(self) -> bool:
        return self._sgl_kernel_backend is not None

    def sgl_kernel_prefill(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, KVPageState]:
        assert self._sgl_kernel_backend is not None
        attn_out, page_state = self._sgl_kernel_backend.prefill(q, k, v, None, causal=True)
        self._sgl_kernel_page_state = page_state
        return attn_out, page_state

    def sgl_kernel_decode(self, q: torch.Tensor) -> torch.Tensor:
        assert self._sgl_kernel_backend is not None and self._sgl_kernel_page_state is not None
        return self._sgl_kernel_backend.decode(q, self._sgl_kernel_page_state, causal=True)

    def _maybe_get_prefill_cache(self, prompt_ids: List[int]) -> tuple[int | None, Any | None]:
        return self.prefill_cache.maybe_get(prompt_ids, self.cache_stats)

    def _maybe_get_prefix_cache(self, prompt_ids: List[int]) -> tuple[Tuple[int, ...], Any] | None:
        return self.prefix_cache.maybe_get(prompt_ids, self.cache_stats)

    def longest_prefix_match_length(self, prompt_ids: List[int]) -> int:
        """Best-effort length of the longest cached prefix without mutating caches."""
        return self.prefix_cache.match_length(prompt_ids)

    def _update_prefill_cache(
        self, prompt_ids: List[int], first_token_id: int, kv_cache: Any
    ) -> None:
        self.prefill_cache.update(prompt_ids, first_token_id, kv_cache)

    def _update_prefix_cache(self, prompt_ids: List[int], kv_cache: Any) -> None:
        self.prefix_cache.update(prompt_ids, kv_cache)

    def _prefill_run(
        self, token_ids: List[int], past_key_values: Any | None, use_context: bool = True
    ):
        """Run prefill with optional chunking to reduce peak memory for long prompts."""

        if len(token_ids) == 0:
            raise ValueError("prefill_run requires at least one token")

        # If dynamic length capture is allowed, bind graph length to the first prompt
        # (bounded by PREFILL_GRAPH_MAX_LEN); otherwise keep disabled.
        if (
            self.enable_cuda_graph
            and self._prefill_graph_dynamic_len
            and self.prefill_graph_seq_len <= 0
        ):
            if len(token_ids) <= self.prefill_graph_max_len:
                self.prefill_graph_seq_len = len(token_ids)
                self._prefill_graph_dynamic_len = False
                logger.info(
                    "Binding CUDA graph prefill seq_len to first prompt len=%d",
                    self.prefill_graph_seq_len,
                )
            else:
                logger.info(
                    "Skipping CUDA graph capture; prompt len=%d exceeds PREFILL_GRAPH_MAX_LEN=%d",
                    len(token_ids),
                    self.prefill_graph_max_len,
                )
                self.enable_cuda_graph = False

        use_graph = (
            self.enable_cuda_graph
            and self.prefill_graph_seq_len > 0
            and len(token_ids) == self.prefill_graph_seq_len
            and past_key_values is None
            and use_context
        )
        if use_graph:
            logits, kv_cache = self._prefill_run_with_graph(token_ids)
            if logits is not None and kv_cache is not None:
                return logits, kv_cache

        if self.chunked_prefill_enabled and len(token_ids) > self.prefill_chunk_size:
            kv_cache = past_key_values
            logits = None
            logger.info(
                "Chunked prefill enabled | total_tokens=%d chunk_size=%d",
                len(token_ids),
                self.prefill_chunk_size,
            )
            for start in range(0, len(token_ids), self.prefill_chunk_size):
                end = start + self.prefill_chunk_size
                chunk = torch.as_tensor(token_ids[start:end], device=self.device).unsqueeze(0)
                outputs = self._prefill_call(
                    chunk, past_key_values=kv_cache, use_context=use_context
                )
                logits = outputs.logits[:, -1, :]
                kv_cache = outputs.past_key_values
            assert logits is not None
            return logits, kv_cache

        input_ids = torch.as_tensor(token_ids, device=self.device).unsqueeze(0)
        outputs = self._prefill_call(
            input_ids, past_key_values=past_key_values, use_context=use_context
        )
        return outputs.logits[:, -1, :], outputs.past_key_values

    # ------------------------------------------------------------------
    # CUDA graph helpers (prefill only; fixed shapes)
    # ------------------------------------------------------------------
    def _prefill_run_with_graph(self, token_ids: List[int]):
        if self._prefill_graph is None:
            if not torch.cuda.is_available():
                return None, None
            if len(token_ids) != self.prefill_graph_seq_len:
                return None, None
            try:
                # Static input for capture; values are overwritten per replay.
                self._prefill_graph_input = torch.empty(
                    (1, self.prefill_graph_seq_len), device=self.device, dtype=torch.long
                )
                self._prefill_graph_input[0, : len(token_ids)] = torch.as_tensor(
                    token_ids, device=self.device
                )
                # Warmup once to prime CUDA memory pools.
                with torch.no_grad():
                    self.model(
                        input_ids=self._prefill_graph_input,
                        past_key_values=None,
                        use_cache=True,
                    )
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = self.model(
                        input_ids=self._prefill_graph_input,
                        past_key_values=None,
                        use_cache=True,
                    )
                    self._prefill_graph_logits = outputs.logits
                    self._prefill_graph_kv = outputs.past_key_values
                self._prefill_graph = graph
                logger.info(
                    "Captured CUDA graph for prefill | seq_len=%d", self.prefill_graph_seq_len
                )
            except Exception as exc:
                logger.warning(
                    "CUDA graph capture failed; disabling graph path (%s)", exc
                )
                self.enable_cuda_graph = False
                self._prefill_graph = None
                return None, None

        if self._prefill_graph is None or self._prefill_graph_input is None:
            return None, None

        self._prefill_graph_input[0, : len(token_ids)] = torch.as_tensor(
            token_ids, device=self.device
        )
        try:
            self._prefill_graph.replay()
        except Exception as exc:
            logger.warning("CUDA graph replay failed; disabling graph path (%s)", exc)
            self.enable_cuda_graph = False
            self._prefill_graph = None
            return None, None
        return self._prefill_graph_logits, self._prefill_graph_kv

    def cap_max_new_tokens(self, prompt_len: int, requested: int | None) -> int | None:
        """Cap max_new_tokens to fit within the model's context window (best-effort)."""
        if requested is None or not self.aggressive_max_new_tokens:
            return requested
        if self.max_context_length is None:
            return requested
        margin = max(0, self.max_context_margin)
        safe_budget = self.max_context_length - prompt_len - margin
        safe_budget = max(1, safe_budget)
        if requested > safe_budget:
            logger.info(
                "Capping max_new_tokens from %d to %d to fit context window "
                "(prompt_len=%d, max_ctx=%d, margin=%d)",
                requested,
                safe_budget,
                prompt_len,
                self.max_context_length,
                margin,
            )
            return safe_budget
        return requested

    def _maybe_init_static_cache(self, prompt_len: int) -> StaticCache | None:
        if not self.enable_static_kv:
            return None
        cache_dtype = self._cache_dtype()
        if self._static_cache is not None:
            cached_dtype = self._cache_values_dtype(self._static_cache)
            if cached_dtype == cache_dtype:
                coerced = self._coerce_static_cache_dtype(self._static_cache, cache_dtype)
                if coerced is not None:
                    self._static_cache = coerced
                    return self._static_cache
            if cached_dtype is None:
                # Cache was never populated; coerce and reuse.
                coerced = self._coerce_static_cache_dtype(self._static_cache, cache_dtype)
                if coerced is not None:
                    self._static_cache = coerced
                    return self._static_cache
            logger.info(
                "Reinitializing StaticCache for dtype change (cache=%s expected=%s)",
                cached_dtype,
                cache_dtype,
            )
            self._static_cache = None
        max_len = self.static_kv_max_len
        if max_len <= 0:
            max_len = self.max_context_length or (prompt_len + self._static_kv_budget)
        if max_len <= 0:
            # As a last resort, disable static cache if we cannot determine a safe length.
            self.enable_static_kv = False
            return None
        try:
            device = torch.device(self.device)
            self._static_cache = StaticCache(
                config=self.model.config, max_cache_len=max_len, device=device, dtype=cache_dtype
            )
            created_dtype = self._cache_values_dtype(self._static_cache)
            if created_dtype is None:
                coerced = self._coerce_static_cache_dtype(self._static_cache, cache_dtype)
                if coerced is None:
                    self.enable_static_kv = False
                    self._static_cache = None
                    return None
                self._static_cache = coerced
                return self._static_cache
            if created_dtype != cache_dtype:
                logger.warning(
                    "StaticCache dtype mismatch after init (cache=%s expected=%s); disabling static KV",
                    created_dtype,
                    cache_dtype,
                )
                self.enable_static_kv = False
                self._static_cache = None
                return None
            logger.info("Initialized StaticCache | max_cache_len=%d", max_len)
            return self._static_cache
        except TypeError:
            # Older transformers may not accept dtype/device; best-effort fallback with validation.
            try:
                self._static_cache = StaticCache(
                    config=self.model.config, max_cache_len=max_len
                )
                cached_dtype = self._cache_values_dtype(self._static_cache)
                if cached_dtype is None:
                    coerced = self._coerce_static_cache_dtype(self._static_cache, cache_dtype)
                    if coerced is None:
                        self.enable_static_kv = False
                        self._static_cache = None
                        return None
                    self._static_cache = coerced
                    return self._static_cache
                if cached_dtype is not None and cached_dtype != cache_dtype:
                    logger.warning(
                        "StaticCache dtype mismatch (cache=%s expected=%s); disabling static KV",
                        cached_dtype,
                        cache_dtype,
                    )
                    self.enable_static_kv = False
                    self._static_cache = None
                    return None
                return self._static_cache
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Failed to init StaticCache; disabling static KV (%s)", exc)
                self.enable_static_kv = False
                self._static_cache = None
                return None
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("Failed to init StaticCache; disabling static KV (%s)", exc)
            self.enable_static_kv = False
            self._static_cache = None
            return None

    def insert_prefix(self, prompt: str) -> None:
        """Prefill and store a prompt's KV in the prefix cache for later reuse."""
        if not self.enable_prefix_cache or self.prefix_cache_size <= 0:
            logger.info("Prefix cache disabled; insert_prefix is a no-op")
            return
        prompt_ids = self.tokenize(prompt)
        if len(prompt_ids) == 0:
            return
        if len(prompt_ids) > self.prefix_cache_max_tokens:
            logger.info(
                "Skipping insert_prefix; prompt too long (len=%d > %d)",
                len(prompt_ids),
                self.prefix_cache_max_tokens,
            )
            return
        logger.info("Inserting prefix into cache | tokens=%d", len(prompt_ids))
        _, kv_cache = self.prefill_forward(prompt_ids)
        # prefill_forward already updates prefix cache with kv_cache

    def cache_metrics(self) -> Dict[str, int]:
        """Return lightweight cache stats."""
        return self.cache_stats.as_dict()

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
