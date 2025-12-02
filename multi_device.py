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
