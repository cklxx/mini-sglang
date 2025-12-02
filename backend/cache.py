"""Cache utilities for mini-sglang backend."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CacheStats:
    def __init__(self) -> None:
        self.prefill_hits = 0
        self.prefill_misses = 0
        self.prefix_hits = 0
        self.prefix_misses = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "prefill_hits": self.prefill_hits,
            "prefill_misses": self.prefill_misses,
            "prefix_hits": self.prefix_hits,
            "prefix_misses": self.prefix_misses,
        }


class _PrefixTrieNode:
    __slots__ = ("children", "kv_cache", "tokens")

    def __init__(self) -> None:
        self.children: Dict[int, "_PrefixTrieNode"] = {}
        self.kv_cache: Any | None = None
        self.tokens: Tuple[int, ...] | None = None

    def insert(self, tokens: Tuple[int, ...], kv_cache: Any) -> None:
        node = self
        for tok in tokens:
            node = node.children.setdefault(tok, _PrefixTrieNode())
        node.kv_cache = kv_cache
        node.tokens = tokens

    def remove(self, tokens: Tuple[int, ...]) -> None:
        path: list[tuple[int, _PrefixTrieNode]] = []
        node = self
        for tok in tokens:
            if tok not in node.children:
                return
            path.append((tok, node))
            node = node.children[tok]
        node.kv_cache = None
        node.tokens = None
        for tok, parent in reversed(path):
            child = parent.children[tok]
            if child.kv_cache is not None or child.children:
                break
            parent.children.pop(tok, None)

    def longest_prefix(self, prompt_ids: List[int]) -> Optional[Tuple[Tuple[int, ...], Any]]:
        node = self
        best_tokens: Tuple[int, ...] | None = None
        best_kv: Any | None = None
        for tok in prompt_ids:
            if tok not in node.children:
                break
            node = node.children[tok]
            if node.kv_cache is not None and node.tokens is not None:
                best_tokens = node.tokens
                best_kv = node.kv_cache
        if best_tokens is None or best_kv is None:
            return None
        return best_tokens, best_kv


class PrefillCache:
    """LRU prefill cache with optional token budget."""

    def __init__(self, size: int, token_budget: int) -> None:
        self.size = size
        self.token_budget = token_budget
        self.cache: OrderedDict[str, Tuple[int, Any, int]] = OrderedDict()

    def maybe_get(self, prompt_ids: List[int], stats: CacheStats) -> tuple[int | None, Any | None]:
        if self.size <= 0:
            return None, None
        prompt_key = ",".join(str(i) for i in prompt_ids)
        if prompt_key not in self.cache:
            stats.prefill_misses += 1
            return None, None
        first_token_id, cached_kv, prompt_len = self.cache.pop(prompt_key)
        self.cache[prompt_key] = (first_token_id, cached_kv, prompt_len)
        stats.prefill_hits += 1
        logger.info("Prefill cache hit for seq_len=%d (first_token_id=%d)", len(prompt_ids), first_token_id)
        return int(first_token_id), cached_kv

    def update(self, prompt_ids: List[int], first_token_id: int, kv_cache: Any) -> None:
        if self.size <= 0:
            return
        prompt_key = ",".join(str(i) for i in prompt_ids)
        self.cache[prompt_key] = (first_token_id, kv_cache, len(prompt_ids))
        self._trim()

    def _trim(self) -> None:
        while self.size > 0 and len(self.cache) > self.size:
            self.cache.popitem(last=False)
        if self.token_budget <= 0:
            return
        total_tokens = sum(length for _, _, length in self.cache.values())
        while self.cache and total_tokens > self.token_budget:
            _, _, length = self.cache.popitem(last=False)
            total_tokens -= length


class PrefixCache:
    """Prefix cache with LRU + token budget and radix lookup."""

    def __init__(
        self,
        enable: bool,
        size: int,
        max_tokens: int,
        token_budget: int,
    ) -> None:
        self.enable = enable
        self.size = size
        self.max_tokens = max_tokens
        self.token_budget = token_budget
        self.cache: OrderedDict[Tuple[int, ...], Any] = OrderedDict()
        self._trie = _PrefixTrieNode()

    def maybe_get(self, prompt_ids: List[int], stats: CacheStats) -> tuple[Tuple[int, ...], Any] | None:
        if not (self.enable and self.size > 0):
            return None
        match = self._trie.longest_prefix(prompt_ids)
        if match is None:
            stats.prefix_misses += 1
            return None
        tokens, cached_kv = match
        stats.prefix_hits += 1
        if tokens in self.cache:
            self.cache.pop(tokens)
            self.cache[tokens] = cached_kv
        return tokens, cached_kv

    def match_length(self, prompt_ids: List[int]) -> int:
        if not (self.enable and self.size > 0):
            return 0
        if not prompt_ids:
            return 0
        match = self._trie.longest_prefix(prompt_ids)
        if match is None:
            return 0
        tokens, _ = match
        return len(tokens)

    def update(self, prompt_ids: List[int], kv_cache: Any) -> None:
        if not (self.enable and self.size > 0):
            return
        if len(prompt_ids) > self.max_tokens:
            logger.debug(
                "Skipping prefix cache insert (len=%d > PREFIX_CACHE_MAX_TOKENS=%d)",
                len(prompt_ids),
                self.max_tokens,
            )
            return
        token_tuple = tuple(prompt_ids)
        self.cache[token_tuple] = kv_cache
        self._trie.insert(token_tuple, kv_cache)
        if len(self.cache) > self.size:
            evicted, _ = self.cache.popitem(last=False)
            self._trie.remove(evicted)
        self._trim_token_budget()

    def insert_prefix(self, prompt_ids: List[int], kv_cache: Any) -> None:
        self.update(prompt_ids, kv_cache)

    def _trim_token_budget(self) -> None:
        if self.token_budget <= 0:
            return
        total_tokens = sum(len(t) for t in self.cache.keys())
        while self.cache and total_tokens > self.token_budget:
            oldest_tokens, _ = self.cache.popitem(last=False)
            total_tokens -= len(oldest_tokens)
            self._trie.remove(oldest_tokens)
