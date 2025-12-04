"""Simple timing helper for consistent duration measurement."""

from __future__ import annotations

import time
from typing import Dict


class Timer:
    """Lightweight named timer with mark/span helpers."""

    def __init__(self) -> None:
        self.marks: Dict[str, float] = {}
        self.mark("start")

    def mark(self, name: str) -> float:
        now = time.perf_counter()
        self.marks[name] = now
        return now

    def span(self, start: str, end: str | None = None) -> float:
        if start not in self.marks:
            return 0.0
        end_time = self.marks.get(end) if end is not None else time.perf_counter()
        if end_time is None:
            return 0.0
        return end_time - self.marks[start]
