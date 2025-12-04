"""MLX-powered backend for MPS generation."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

try:  # Lazy import so the module can be imported on non-MLX setups.
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.generate import generate_step
    from mlx_lm.models import cache as mlx_cache
    from mlx_lm.utils import _download as mlx_download
    from mlx_lm.utils import load_model as mlx_load_model
    from mlx_lm.utils import load_tokenizer as mlx_load_tokenizer
except Exception as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "MlxBackend requires the `mlx` and `mlx-lm` packages. "
        "Install with `pip install mlx mlx-lm` on Apple Silicon."
    ) from exc

_default_max_new_tokens = 512
try:
    from config import MAX_NEW_TOKENS_DEFAULT as _config_max_new_tokens
except Exception:  # pragma: no cover - fallback if config import fails
    _config_max_new_tokens = _default_max_new_tokens
DEFAULT_MAX_NEW_TOKENS = _config_max_new_tokens


class MlxBackend:
    """Backend that wraps mlx-lm generation for SGLangMiniEngine."""

    def __init__(self, model_name: str, device: str = "mps") -> None:
        self.model_name = model_name
        self.device = device or "mps"
        self.max_new_tokens_default = DEFAULT_MAX_NEW_TOKENS
        self.aggressive_max_new_tokens = os.getenv("AGGRESSIVE_MAX_NEW_TOKENS", "1") != "0"
        self.max_context_margin = int(os.getenv("MAX_CONTEXT_MARGIN", "16"))
        self.prefill_step_size = int(os.getenv("MLX_PREFILL_STEP_SIZE", "2048"))
        self.max_kv_size = int(os.getenv("MLX_MAX_KV_SIZE", "0"))
        self._planned_max_tokens: Optional[int] = None
        self.trust_remote_code = os.getenv("MLX_TRUST_REMOTE_CODE", "0") != "0"
        self.mlx_cache_dir = Path(
            os.getenv("MLX_CACHE_DIR", Path.home() / ".cache" / "mini_sglang" / "mlx")
        )

        try:  # Prefer GPU/MPS execution when available.
            mx.set_default_device(mx.Device(mx.gpu))
        except Exception as exc:  # pragma: no cover - best effort
            logger.debug("Failed to set MLX default device to GPU/MPS: %s", exc)

        logger.info("Loading MLX backend model=%s", model_name)
        model, tokenizer, cfg = self._load_or_convert(model_name)
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = cfg

        self.max_context_length = cfg.get("max_position_embeddings")
        eos_candidates = []
        tok_eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if tok_eos_ids:
            eos_candidates = list(tok_eos_ids)
        else:
            tok_eos = getattr(self.tokenizer, "eos_token_id", None)
            if tok_eos is not None:
                eos_candidates = [tok_eos]
        self.eos_token_id = int(eos_candidates[0] if eos_candidates else 0)

        logger.info(
            "MLX backend ready: device=%s eos_token_id=%d max_ctx=%s",
            self.device,
            self.eos_token_id,
            self.max_context_length,
        )

    def prepare_generation(self, max_new_tokens: Optional[int]) -> None:
        self._planned_max_tokens = max_new_tokens

    def tokenize(self, prompt: str) -> List[int]:
        encode_fn = cast(Callable[..., List[int]], self.tokenizer.encode)
        return encode_fn(prompt, add_special_tokens=False)

    def decode_tokens(self, token_ids: List[int]) -> str:
        decode_fn = cast(Callable[..., str], self.tokenizer.decode)
        return decode_fn(token_ids, skip_special_tokens=True)

    def cap_max_new_tokens(self, prompt_len: int, requested: int | None) -> int | None:
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

    def prefill_forward(self, prompt_ids: List[int], use_context: bool = True) -> Tuple[int, Any]:
        prompt_arr = mx.array(prompt_ids, dtype=mx.uint32)
        max_tokens = self._planned_max_tokens or self.max_new_tokens_default
        max_tokens = max(1, max_tokens)
        logger.info(
            "Running MLX prefill with seq_len=%d max_new_tokens=%d on device=%s",
            len(prompt_ids),
            max_tokens,
            self.device,
        )

        prompt_cache = self._make_prompt_cache()
        token_gen = self._make_generator(prompt_arr, prompt_cache, max_tokens)

        try:
            first_token, _ = next(token_gen)
        except StopIteration:
            logger.warning("MLX generator produced no tokens; emitting EOS")
            return self.eos_token_id, {"generator": None, "tokens": []}

        kv_state = {
            "generator": token_gen,
            "prompt_cache": prompt_cache,
            "tokens": [int(first_token)],
        }
        logger.info(
            "MLX prefill produced first_token_id=%d (text=%r)",
            int(first_token),
            self.decode_tokens([int(first_token)]),
        )
        return int(first_token), kv_state

    def decode_forward(
        self, last_token_id: int, kv_cache: Any, use_context: bool = True
    ) -> Tuple[int, Any]:
        gen = kv_cache.get("generator")
        if gen is None:
            logger.info("MLX generator exhausted; returning EOS")
            return self.eos_token_id, kv_cache
        try:
            next_token, _ = next(gen)
        except StopIteration:
            logger.info("MLX generator reached StopIteration; returning EOS")
            next_token = self.eos_token_id
            kv_cache["generator"] = None
        next_token = int(next_token)
        kv_cache.setdefault("tokens", []).append(next_token)
        return next_token, kv_cache

    def longest_prefix_match_length(self, prompt_ids: List[int]) -> int:
        return 0

    def insert_prefix(self, prompt: str) -> None:
        logger.debug("MLX backend does not support prefix warming; skipping insert_prefix")

    def generate_streaming_baseline(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        log_stride: int = 32,
        stream_callback: Optional[Any] = None,
    ) -> Tuple[str, float]:
        prompt_tokens = [int(t) for t in prompt_ids]
        start = time.perf_counter()
        chunks: list[str] = []
        for idx, resp in enumerate(
            stream_generate(
                self.model,
                self.tokenizer,
                prompt_tokens,
                max_tokens=max_new_tokens,
            ),
            1,
        ):
            text_delta = resp.text
            if stream_callback:
                stream_callback(text_delta)
            chunks.append(text_delta)
            if idx == 1 or idx % max(1, log_stride) == 0:
                logger.info("[mlx-stream] chunk %03d: %r", idx, text_delta)
        duration = time.perf_counter() - start
        return "".join(chunks), duration

    def _load_or_convert(self, model_name: str):
        try:
            return self._load_model(model_name, strict=True)
        except ValueError as exc:
            logger.info(
                "MLX strict load failed for %s (%s); retrying with strict=False",
                model_name,
                exc,
            )
            return self._load_model(model_name, strict=False)

    def _load_model(self, model_path: str, strict: bool):
        local_path = mlx_download(model_path)
        model, cfg = mlx_load_model(
            local_path,
            lazy=False,
            strict=strict,
            model_config={"trust_remote_code": self.trust_remote_code},
        )
        tokenizer = mlx_load_tokenizer(
            local_path,
            tokenizer_config_extra={"trust_remote_code": self.trust_remote_code},
            eos_token_ids=cfg.get("eos_token_id", None),
        )
        return model, tokenizer, cfg

    def _make_prompt_cache(self) -> Any:
        if self.max_kv_size > 0:
            return mlx_cache.make_prompt_cache(self.model, max_kv_size=self.max_kv_size)
        return mlx_cache.make_prompt_cache(self.model)

    def _make_generator(
        self, prompt: mx.array, prompt_cache: Any, max_tokens: int
    ) -> Generator[Tuple[Any, Any], None, None]:
        return generate_step(
            prompt=prompt,
            model=self.model,
            max_tokens=max_tokens,
            prompt_cache=prompt_cache,
            prefill_step_size=self.prefill_step_size,
        )
