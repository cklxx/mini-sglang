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
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import httpx

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from backend.cache import CacheStats, PrefixCache, PrefillCache
from optimizations import configure_torch, inference_context, maybe_compile_model


logger = logging.getLogger(__name__)


def resolve_model_path(model_name: str) -> str:
    """Return a local model path, falling back to ModelScope when HF is blocked.

    The flow prefers an existing local Hugging Face cache, then tries a quick
    HEAD check + download from huggingface.co. If that fails (e.g., network
    egress is blocked), it falls back to ModelScope ("魔搭") using the same
    model id or ``MODELSCOPE_MODEL_NAME`` if provided.
    """

    local_override = os.getenv("MODEL_LOCAL_DIR")
    if local_override:
        logger.info("Using MODEL_LOCAL_DIR=%s", local_override)
        return local_override

    # 1) Use cached Hugging Face files if they already exist to avoid network.
    try:
        from huggingface_hub import snapshot_download as hf_snapshot_download

        cached_path = hf_snapshot_download(repo_id=model_name, local_files_only=True)
        logger.info("Using cached Hugging Face model at %s", cached_path)
        return cached_path
    except Exception:
        cached_path = None

    # 2) Try Hugging Face if reachable.
    hf_ok = False
    try:
        resp = httpx.head(
            f"https://huggingface.co/{model_name}",
            follow_redirects=True,
            timeout=3.0,
        )
        hf_ok = resp.status_code < 400
        if not hf_ok:
            logger.warning(
                "Hugging Face responded with status %s for %s", resp.status_code, model_name
            )
    except Exception as err:  # pragma: no cover - best-effort network probe
        logger.warning("Hugging Face connectivity check failed: %s", err)

    if hf_ok:
        try:
            from huggingface_hub import snapshot_download as hf_snapshot_download

            hf_path = hf_snapshot_download(repo_id=model_name, local_files_only=False)
            logger.info("Downloaded model from Hugging Face to %s", hf_path)
            return hf_path
        except Exception as err:
            logger.warning("Hugging Face download failed (%s); trying ModelScope.", err)

    # 3) Fallback to ModelScope (魔搭)
    ms_repo = os.getenv("MODELSCOPE_MODEL_NAME", model_name)
    try:
        from modelscope import snapshot_download as ms_snapshot_download

        ms_path = ms_snapshot_download(ms_repo)
        logger.info("Downloaded model from ModelScope repo=%s to %s", ms_repo, ms_path)
        return ms_path
    except Exception as err:
        logger.error(
            "Model download failed from Hugging Face%s and ModelScope",
            " (cache hit)" if cached_path else "",
            exc_info=err,
        )
        raise


class ModelBackend:
    """Backend that handles model/tokenizer loading and forward passes."""

    def __init__(self, model_name: str, device: str, compile_model: bool = True) -> None:
        configure_torch(device)
        model_path = resolve_model_path(model_name)

        self.device = device
        compile_enabled = self._flag_from_env("COMPILE_MODEL", default=compile_model)
        self.tensor_parallel_size = self._resolve_tensor_parallel_size()
        use_tensor_parallel = self._can_use_tensor_parallel()
        model_kwargs = self._tensor_parallel_kwargs(use_tensor_parallel)
        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        self.tokenizer = self._load_tokenizer(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
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
        self._init_cuda_graph_settings()
        self._init_generation_config()

        logger.info(
            "Model backend ready: model=%s device=%s eos_token_id=%s", model_name, device, self.eos_token_id
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

    def _get_prefill_graph(self, seq_len: int) -> Callable[[torch.Tensor], Any] | None:
        if not self.cuda_graph_enabled or seq_len > self.cuda_graph_max_seq_len:
            return None
        if seq_len in self._prefill_graphs:
            graph = self._prefill_graphs.pop(seq_len)
            self._prefill_graphs[seq_len] = graph
            return graph
        sample_ids = torch.zeros((1, seq_len), device=self.device, dtype=torch.long)

        def _prefill_with_cache(input_ids: torch.Tensor) -> Any:
            return self.model(input_ids=input_ids, use_cache=True)

        try:
            graph_callable = torch.cuda.make_graphed_callables(_prefill_with_cache, (sample_ids,))
            graph = cast(Callable[[torch.Tensor], Any], graph_callable)
            self._prefill_graphs[seq_len] = graph
            if len(self._prefill_graphs) > 4:
                self._prefill_graphs.pop(next(iter(self._prefill_graphs)))
            logger.info("Captured CUDA graph for prefill seq_len=%d", seq_len)
            return graph
        except Exception as exc:  # pragma: no cover - best-effort path
            logger.warning("Failed to capture CUDA graph for seq_len=%d (%s)", seq_len, exc)
            self.cuda_graph_enabled = False
            return None

    def _prefill_call(self, input_ids: torch.Tensor, past_key_values: Any | None):
        graph = None if past_key_values is not None else self._get_prefill_graph(input_ids.shape[1])
        auto_ctx, inf_ctx = inference_context(self.device)
        with auto_ctx, inf_ctx:
            if graph is not None:
                return graph(input_ids)
            return self.model(
                input_ids=input_ids, past_key_values=past_key_values, use_cache=True
            )

    def _decode_call(self, input_ids: torch.Tensor, past_key_values: Any) -> Any:
        """Decode step with optional CUDA graph capture (single-token decode)."""
        if not self.decode_graph_enabled:
            return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        if self._decode_graph is None:
            try:
                sample_ids = torch.zeros_like(input_ids)

                def _decode(ids: torch.Tensor, kv: Any):
                    return self.model(input_ids=ids, past_key_values=kv, use_cache=True)

                self._decode_graph = torch.cuda.make_graphed_callables(_decode, (sample_ids, past_key_values))
                logger.info("Captured CUDA graph for decode step")
            except Exception as exc:  # pragma: no cover - best-effort path
                logger.warning("Failed to capture decode CUDA graph (%s)", exc)
                self.decode_graph_enabled = False
                return self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return cast(Callable[[torch.Tensor, Any], Any], self._decode_graph)(input_ids, past_key_values)

    def prefill_forward(self, prompt_ids: List[int]) -> Tuple[int, Any]:
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
            logits, kv_cache = self._prefill_run(remaining_ids, past_key_values=cached_kv)
        else:
            logits, kv_cache = self._prefill_run(prompt_ids, past_key_values=None)
        first_token_id = int(torch.argmax(logits, dim=-1).item())

        self._update_prefill_cache(prompt_ids, first_token_id, kv_cache)
        self._update_prefix_cache(prompt_ids, kv_cache)
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
            if self.decode_graph_enabled:
                outputs = self._decode_call(input_ids, kv_cache)
            else:
                outputs = self.model(
                    input_ids=input_ids, past_key_values=kv_cache, use_cache=True
                )
        logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(logits, dim=-1).item()
        new_kv_cache = outputs.past_key_values
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
                "Requested tensor parallelism (TENSOR_PARALLEL_SIZE=%d) but found only %d CUDA devices",
                self.tensor_parallel_size,
                torch.cuda.device_count(),
            )
        return {}

    def _resolve_torch_dtype(self) -> Optional[torch.dtype]:
        """Optional dtype override for model weights."""
        dtype_str = os.getenv("MODEL_DTYPE") or os.getenv("TORCH_DTYPE")
        if dtype_str is None:
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

    def _load_tokenizer(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _init_caches(self) -> None:
        # Simple LRU caches for tokenization and prefill KV reuse.
        self.token_cache_size = int(os.getenv("TOKEN_CACHE_SIZE", "32"))
        self.prefill_cache_size = int(os.getenv("PREFILL_CACHE_SIZE", "8"))
        self.prefill_cache_token_budget = int(os.getenv("PREFILL_CACHE_TOKEN_BUDGET", "65536"))
        self.enable_prefix_cache = self._flag_from_env("PREFIX_CACHE", default=True)
        self.prefix_cache_size = int(os.getenv("PREFIX_CACHE_SIZE", "16"))
        self.prefix_cache_max_tokens = int(os.getenv("PREFIX_CACHE_MAX_TOKENS", "4096"))
        self.prefix_cache_token_budget = int(os.getenv("PREFIX_CACHE_TOKEN_BUDGET", "65536"))
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
        )

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

    def _init_cuda_graph_settings(self) -> None:
        self.cuda_graph_enabled = (
            self._flag_from_env("ENABLE_CUDA_GRAPH", default=True)
            and self.device.startswith("cuda")
        )
        self.cuda_graph_max_seq_len = int(os.getenv("CUDA_GRAPH_MAX_SEQ_LEN", "512"))
        self._prefill_graphs: Dict[int, Callable[[torch.Tensor], Any]] = {}
        self.decode_graph_enabled = (
            self._flag_from_env("ENABLE_DECODE_CUDA_GRAPH", default=True)
            and self.device.startswith("cuda")
        )
        self._decode_graph: Callable[[torch.Tensor, Any], Any] | None = None

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

    def _maybe_get_prefill_cache(self, prompt_ids: List[int]) -> tuple[int | None, Any | None]:
        return self.prefill_cache.maybe_get(prompt_ids, self.cache_stats)

    def _maybe_get_prefix_cache(self, prompt_ids: List[int]) -> tuple[Tuple[int, ...], Any] | None:
        return self.prefix_cache.maybe_get(prompt_ids, self.cache_stats)

    def longest_prefix_match_length(self, prompt_ids: List[int]) -> int:
        """Best-effort length of the longest cached prefix without mutating caches."""
        return self.prefix_cache.match_length(prompt_ids)

    def _update_prefill_cache(self, prompt_ids: List[int], first_token_id: int, kv_cache: Any) -> None:
        self.prefill_cache.update(prompt_ids, first_token_id, kv_cache)

    def _update_prefix_cache(self, prompt_ids: List[int], kv_cache: Any) -> None:
        self.prefix_cache.update(prompt_ids, kv_cache)

    def _prefill_run(self, token_ids: List[int], past_key_values: Any | None):
        """Run prefill with optional chunking to reduce peak memory for long prompts."""

        if len(token_ids) == 0:
            raise ValueError("prefill_run requires at least one token")

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
                outputs = self._prefill_call(chunk, past_key_values=kv_cache)
                logits = outputs.logits[:, -1, :]
                kv_cache = outputs.past_key_values
            assert logits is not None
            return logits, kv_cache

        input_ids = torch.as_tensor(token_ids, device=self.device).unsqueeze(0)
        outputs = self._prefill_call(input_ids, past_key_values=past_key_values)
        return outputs.logits[:, -1, :], outputs.past_key_values

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
