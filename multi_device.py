"""Multi-device engine pool for simple round-robin dispatch."""

from __future__ import annotations

import logging
import os
import threading
import random
from typing import Any

import torch

from config import MAX_NEW_TOKENS_DEFAULT, MODEL_NAME, get_device
from optimizations import warmup_engine

logger = logging.getLogger(__name__)


class EnginePool:
    """Maintain a pool of engines across available devices (CUDA-first)."""

    def __init__(
        self,
        *,
        ModelBackend,
        SGLangMiniEngine,
        model_name: str = MODEL_NAME,
        max_new_tokens_default: int = MAX_NEW_TOKENS_DEFAULT,
        compile_model: bool = False,
    ) -> None:
        enable_multi = os.getenv("ENABLE_MULTI_DEVICE", "1") != "0"
        devices: list[str]
        if enable_multi and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
            logger.info("Initializing multi-device pool on %s", devices)
        else:
            devices = [get_device()]
            logger.info("Initializing single-device pool on %s", devices)

        self.engines: list[Any] = []
        for dev in devices:
            backend = ModelBackend(model_name=model_name, device=dev, compile_model=compile_model)
            engine = SGLangMiniEngine(
                backend=backend, max_new_tokens_default=max_new_tokens_default
            )
            self.engines.append(engine)
        self._inflight = [0 for _ in self.engines]

        if not self.engines:
            raise RuntimeError("EnginePool could not initialize any engines")

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._next = 0
        self.max_inflight_total = int(os.getenv("MAX_INFLIGHT_TOTAL", "0"))
        self.max_inflight_per_engine = int(os.getenv("MAX_INFLIGHT_PER_ENGINE", "0"))
        self.adaptive_max_new_tokens = os.getenv("ADAPTIVE_MAX_NEW_TOKENS", "0") != "0"
        self.adaptive_inflight_threshold = int(
            os.getenv("ADAPTIVE_MAX_INFLIGHT_THRESHOLD", str(len(self.engines)))
        )
        self.adaptive_factor = float(os.getenv("ADAPTIVE_MAX_NEW_TOKENS_FACTOR", "0.8"))
        self.scheduler_mode = os.getenv("SCHEDULER_MODE", "rr").lower()
        valid_modes = {"rr", "fsfs", "random", "cache_aware"}
        if self.scheduler_mode not in valid_modes:
            logger.warning(
                "Invalid SCHEDULER_MODE=%s, falling back to round robin", self.scheduler_mode
            )
            self.scheduler_mode = "rr"

    @property
    def primary_backend(self) -> Any:
        return self.engines[0].backend

    def release(self, lease: int) -> None:
        with self._cond:
            if lease < 0 or lease >= len(self._inflight):
                logger.warning("Invalid lease index %s in release", lease)
                return
            self._inflight[lease] -= 1
            if self._inflight[lease] < 0:
                logger.warning("Inflight count negative for engine %d; resetting to 0", lease)
                self._inflight[lease] = 0
            self._cond.notify_all()

    def _pick_round_robin(self, blocked: set[int]) -> int | None:
        n = len(self.engines)
        start = self._next % n
        for offset in range(n):
            idx = (start + offset) % n
            if idx in blocked:
                continue
            self._next = (idx + 1) % n
            return idx
        return None

    def _pick_random(self, blocked: set[int]) -> int | None:
        candidates = [i for i in range(len(self.engines)) if i not in blocked]
        if not candidates:
            return None
        idx = random.choice(candidates)
        self._next = (idx + 1) % len(self.engines)
        return idx

    def _pick_cache_aware(self, prompt_ids: list[int], blocked: set[int]) -> int | None:
        if not prompt_ids:
            return self._pick_round_robin(blocked)

        start = self._next % len(self.engines)
        best_idx = None
        best_match = -1
        best_load = None
        engine_count = len(self.engines)
        for offset in range(engine_count):
            idx = (start + offset) % engine_count
            if idx in blocked:
                continue
            eng = self.engines[idx]
            match_len = eng.backend.longest_prefix_match_length(prompt_ids)
            load = self._inflight[idx]
            if match_len > best_match or (match_len == best_match and (best_load is None or load < best_load)):
                best_match = match_len
                best_idx = idx
                best_load = load

        if best_idx is None or best_match <= 0:
            return self._pick_round_robin(blocked)

        self._next = (best_idx + 1) % engine_count
        return best_idx

    def _pick_fsfs(self, blocked: set[int]) -> int | None:
        candidates = [(idx, load) for idx, load in enumerate(self._inflight) if idx not in blocked]
        if not candidates:
            return None
        min_load = min(load for _, load in candidates)
        for idx, load in candidates:
            if load == min_load:
                return idx
        return None

    def pick(self, prompt_ids: list[int] | None = None) -> tuple[Any, int]:
        with self._cond:
            while True:
                if self.max_inflight_total > 0 and sum(self._inflight) >= self.max_inflight_total:
                    self._cond.wait()
                    continue

                blocked: set[int] = set()
                if self.max_inflight_per_engine > 0:
                    blocked = {
                        idx for idx, load in enumerate(self._inflight) if load >= self.max_inflight_per_engine
                    }

                if self.scheduler_mode == "random":
                    idx = self._pick_random(blocked)
                elif self.scheduler_mode == "cache_aware":
                    idx = self._pick_cache_aware(prompt_ids or [], blocked)
                elif self.scheduler_mode == "fsfs":
                    idx = self._pick_fsfs(blocked)
                else:
                    idx = self._pick_round_robin(blocked)

                if idx is None:
                    self._cond.wait()
                    continue

                self._inflight[idx] += 1
                return self.engines[idx], idx

    def warm(self, tokens: int) -> None:
        for eng in self.engines:
            warmup_engine(eng, max_new_tokens=tokens)

    def warm_prefixes(self, prompts: list[str]) -> None:
        if not prompts:
            return
        for eng in self.engines:
            for prompt in prompts:
                try:
                    eng.backend.insert_prefix(prompt)
                except Exception as exc:
                    logger.warning("Failed to warm prefix %r on engine %s (%s)", prompt[:50], eng, exc)

    def metrics(self) -> dict[str, Any]:
        """Lightweight pool metrics for observability."""
        with self._lock:
            inflight = list(self._inflight)
            total_inflight = sum(self._inflight)
            scheduler = self.scheduler_mode
            pool_size = len(self.engines)
        cache_totals = {"prefill_hits": 0, "prefill_misses": 0, "prefix_hits": 0, "prefix_misses": 0}
        for eng in self.engines:
            stats = eng.backend.cache_metrics()
            for k, v in stats.items():
                cache_totals[k] = cache_totals.get(k, 0) + v
        return {
            "scheduler": scheduler,
            "pool_size": pool_size,
            "inflight_total": total_inflight,
            "inflight_per_engine": inflight,
            "cache": cache_totals,
        }

    def adapt_max_new_tokens(self, prompt_len: int, requested: int, backend: Any) -> int:
        """Apply backend cap + optional adaptive downscale under load."""
        capped = backend.cap_max_new_tokens(prompt_len, requested)
        max_tokens = capped if capped is not None else requested
        if not self.adaptive_max_new_tokens:
            return max_tokens
        with self._lock:
            total = sum(self._inflight)
        if self.adaptive_inflight_threshold > 0 and total >= self.adaptive_inflight_threshold:
            factor = max(0.1, self.adaptive_factor)
            new_tokens = max(1, int(max_tokens * factor))
            if new_tokens < max_tokens:
                logger.info(
                    "Adaptive max_new_tokens downscale | from=%d to=%d inflight=%d threshold=%d factor=%.2f",
                    max_tokens,
                    new_tokens,
                    total,
                    self.adaptive_inflight_threshold,
                    factor,
                )
                max_tokens = new_tokens
        return max_tokens
