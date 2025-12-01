"""Multi-device engine pool for simple round-robin dispatch."""

from __future__ import annotations

import logging
import os
import threading
from typing import List, Tuple

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
        devices: List[str]
        if enable_multi and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count())]
            logger.info("Initializing multi-device pool on %s", devices)
        else:
            devices = [get_device()]
            logger.info("Initializing single-device pool on %s", devices)

        self.engines: List[SGLangMiniEngine] = []
        for dev in devices:
            backend = ModelBackend(model_name=model_name, device=dev, compile_model=compile_model)
            engine = SGLangMiniEngine(
                backend=backend, max_new_tokens_default=max_new_tokens_default
            )
            self.engines.append(engine)

        if not self.engines:
            raise RuntimeError("EnginePool could not initialize any engines")

        self._lock = threading.Lock()
        self._next = 0

    @property
    def primary_backend(self):
        return self.engines[0].backend

    def pick(self):
        with self._lock:
            idx = self._next % len(self.engines)
            self._next += 1
        return self.engines[idx]

    def warm(self, tokens: int) -> None:
        for eng in self.engines:
            warmup_engine(eng, max_new_tokens=tokens)
