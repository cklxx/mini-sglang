"""Optional sgl_kernel attention backend (with safe torch fallback)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    import sgl_kernel.flash_attn as _sgl_flash  # type: ignore
    _sgl_import_error: Optional[Exception] = None
except Exception as exc:
    _sgl_flash = None
    _sgl_import_error = exc


def sgl_kernel_available() -> bool:
    return _sgl_flash is not None and torch.cuda.is_available()


def sgl_kernel_unavailable_reason() -> str:
    if _sgl_flash is None:
        return (
            f"import error={_sgl_import_error!r}"
            if "_sgl_import_error" in globals()
            else "import failed"
        )
    if not torch.cuda.is_available():
        return "CUDA not available"
    return "unknown"


@dataclass
class KVPageState:
    k_cache: torch.Tensor  # [seq, num_kv_heads, head_dim]
    v_cache: torch.Tensor  # [seq, num_kv_heads, head_dim]
    page_table: torch.Tensor  # [1, seq] indices for flash kernels
    cache_seqlens: torch.Tensor  # [1] lengths


class SglKernelAttentionBackend(nn.Module):
    """Attention backend that prefers sgl_kernel but falls back to torch SDPA."""

    def __init__(
        self, num_heads: int, head_dim: int, num_kv_heads: int, page_size: int = 512
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self._use_sgl = sgl_kernel_available()
        if self._use_sgl:
            logger.info("sgl_kernel detected; flash_attn_with_kvcache will be used on CUDA")
        else:
            logger.info("sgl_kernel unavailable or CUDA not found; falling back to torch SDPA")

    def prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        page_state: KVPageState | None,
        causal: bool = True,
    ) -> Tuple[torch.Tensor, KVPageState]:
        """Prefill over a prompt: compute attention and seed KV cache."""
        if not self._use_sgl:
            attn_out = scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal
            ).transpose(1, 2)
            attn_out = attn_out.reshape(q.shape[0], q.shape[1], -1)
            k_cache = k.contiguous().squeeze(0)
            v_cache = v.contiguous().squeeze(0)
            page_table = torch.arange(k_cache.shape[0], device=k.device, dtype=torch.int32)[None, :]
            cache_seqlens = torch.tensor([k_cache.shape[0]], device=k.device, dtype=torch.int32)
            return attn_out, KVPageState(
                k_cache=k_cache, v_cache=v_cache, page_table=page_table, cache_seqlens=cache_seqlens
            )

        assert _sgl_flash is not None
        attn_out = _sgl_flash.flash_attn_with_kvcache(
            q=q.reshape(-1, self.num_heads, self.head_dim),
            k_cache=k.reshape(1, k.shape[1], self.num_kv_heads, self.head_dim),
            v_cache=v.reshape(1, v.shape[1], self.num_kv_heads, self.head_dim),
            page_table=None,
            cache_seqlens=None,
            cu_seqlens_q=None,
            cu_seqlens_k_new=None,
            max_seqlen_q=q.shape[1],
            softmax_scale=self.head_dim**-0.5,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
        )
        page_table = torch.arange(k.shape[1], device=k.device, dtype=torch.int32)[None, :]
        cache_seqlens = torch.tensor([k.shape[1]], device=k.device, dtype=torch.int32)
        return attn_out, KVPageState(
            k_cache=k.squeeze(0),
            v_cache=v.squeeze(0),
            page_table=page_table,
            cache_seqlens=cache_seqlens,
        )

    def decode(
        self,
        q: torch.Tensor,
        page_state: KVPageState,
        causal: bool = True,
    ) -> torch.Tensor:
        """Single-token decode using existing KV cache."""
        if not self._use_sgl:
            k_cache = page_state.k_cache
            v_cache = page_state.v_cache
            attn_out = scaled_dot_product_attention(
                q.transpose(1, 2),
                k_cache.unsqueeze(0).transpose(1, 2),
                v_cache.unsqueeze(0).transpose(1, 2),
                is_causal=causal,
            ).transpose(1, 2)
            return attn_out.reshape(q.shape[0], q.shape[1], -1)

        assert _sgl_flash is not None
        o = _sgl_flash.flash_attn_with_kvcache(
            q=q.reshape(-1, self.num_heads, self.head_dim),
            k_cache=page_state.k_cache.reshape(1, -1, self.num_kv_heads, self.head_dim),
            v_cache=page_state.v_cache.reshape(1, -1, self.num_kv_heads, self.head_dim),
            page_table=page_state.page_table,
            cache_seqlens=page_state.cache_seqlens,
            cu_seqlens_q=None,
            cu_seqlens_k_new=None,
            max_seqlen_q=1,
            softmax_scale=self.head_dim**-0.5,
            causal=causal,
            window_size=(-1, -1),
            softcap=0.0,
        )
        return o
