"""RadixAttention-style wrapper that delegates to a backend (sgl_kernel or torch)."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

from backend.sgl_kernel_backend import KVPageState, SglKernelAttentionBackend, sgl_kernel_available


class RadixAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        num_kv_heads: int,
        scaling: Optional[float] = None,
        layer_id: int = 0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.layer_id = layer_id
        self.scaling = scaling or head_dim**-0.5
        self.backend = (
            SglKernelAttentionBackend(num_heads=num_heads, head_dim=head_dim, num_kv_heads=num_kv_heads)
            if sgl_kernel_available()
            else None
        )
        self.page_state: KVPageState | None = None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if self.backend is None:
            # torch fallback
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            out = scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scaling
            ).transpose(1, 2)
            return out

        if self.page_state is None:
            attn_out, state = self.backend.prefill(q, k, v, None, causal=True)
            if save_kv_cache:
                self.page_state = state
            return attn_out

        attn_out = self.backend.decode(q[:, -1:, :, :], self.page_state, causal=True)
        if save_kv_cache:
            new_k = torch.cat([self.page_state.k_cache, k.squeeze(0)], dim=0)
            new_v = torch.cat([self.page_state.v_cache, v.squeeze(0)], dim=0)
            seq_len = new_k.shape[0]
            page_table = torch.arange(seq_len, device=new_k.device, dtype=torch.int32)[None, :]
            cache_seqlens = torch.tensor([seq_len], device=new_k.device, dtype=torch.int32)
            self.page_state = KVPageState(
                k_cache=new_k, v_cache=new_v, page_table=page_table, cache_seqlens=cache_seqlens
            )
        return attn_out
