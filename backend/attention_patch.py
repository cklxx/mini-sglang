"""Lightweight attention patch to route self-attn through sgl_kernel backend.

This targets Llama/Qwen-style attention modules with q_proj/k_proj/v_proj/o_proj
and rotary embedding. It keeps a minimal KV cache compatible with Hugging Face
generation while allowing a backend switch to sgl_kernel flash attention on CUDA
(or torch SDPA fallback on CPU/MPS).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
from torch import nn

from backend.sgl_kernel_backend import KVPageState, SglKernelAttentionBackend

logger = logging.getLogger(__name__)

# Optional sglang components for closer parity with upstream layers.
try:  # pragma: no cover - optional dependency
    from sglang.srt.layers.rotary_embedding import get_rope
except Exception:

    def get_rope(*_: Any, **__: Any) -> None:
        return None


class SglKernelAttentionWrapper(nn.Module):
    """Wrap a HF attention module and route attn math to SglKernelAttentionBackend."""

    def __init__(
        self,
        attn_module: nn.Module,
        backend: SglKernelAttentionBackend,
    ) -> None:
        super().__init__()
        self.attn = attn_module
        self.backend = backend
        self.page_state: KVPageState | None = None
        self.rope = None
        head_dim = getattr(attn_module, "head_dim", None) or getattr(attn_module, "hidden_size", None)
        num_heads = getattr(attn_module, "num_heads", None)
        if head_dim is not None and num_heads is not None:
            try:
                self.rope = get_rope(head_dim, rotary_dim=head_dim, max_position=attn_module.max_position_embeddings)
            except Exception:
                self.rope = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Compute qkv using existing projections
        q = self.attn.q_proj(hidden_states)
        k = self.attn.k_proj(hidden_states)
        v = self.attn.v_proj(hidden_states)
        if hasattr(self.attn, "rotary_emb"):
            q, k = self.attn.rotary_emb(q, k, position_ids)
        elif self.rope is not None:
            # fallback rope if HF attn lacks rotary_emb attribute
            pos = position_ids if position_ids is not None else torch.arange(
                q.shape[1], device=q.device, dtype=torch.long
            ).unsqueeze(0)
            q, k = self.rope(pos, q, k)
        batch, seq, _ = q.shape
        head_dim = q.shape[-1] // self.attn.num_heads
        q = q.view(batch, seq, self.attn.num_heads, head_dim)
        k = k.view(batch, seq, getattr(self.attn, "num_key_value_heads", self.attn.num_heads), head_dim)
        v = v.view(batch, seq, getattr(self.attn, "num_key_value_heads", self.attn.num_heads), head_dim)

        if past_key_value is not None and self.page_state is not None:
            # Decode step
            attn_out = self.backend.decode(q[:, -1:, :, :], self.page_state, causal=True)
            # Append to cache for future decode steps
            new_k = torch.cat([self.page_state.k_cache, k.squeeze(0)], dim=0)
            new_v = torch.cat([self.page_state.v_cache, v.squeeze(0)], dim=0)
            seq_len = new_k.shape[0]
            page_table = torch.arange(seq_len, device=new_k.device, dtype=torch.int32)[None, :]
            cache_seqlens = torch.tensor([seq_len], device=new_k.device, dtype=torch.int32)
            self.page_state = KVPageState(
                k_cache=new_k, v_cache=new_v, page_table=page_table, cache_seqlens=cache_seqlens
            )
        else:
            attn_out, self.page_state = self.backend.prefill(q, k, v, None, causal=True)

        # Project back to hidden size
        attn_out = attn_out.reshape(batch, -1, self.attn.num_heads * head_dim)
        attn_out = self.attn.o_proj(attn_out)

        # Return a dummy past_key_value to satisfy HF; actual cache lives in wrapper
        dummy_k = torch.empty(0, device=hidden_states.device)
        dummy_v = torch.empty(0, device=hidden_states.device)
        return attn_out, (dummy_k, dummy_v)


def patch_model_with_sgl_kernel(model: nn.Module, backend: SglKernelAttentionBackend) -> int:
    """Replace Llama/Qwen-style attention modules with SglKernelAttentionWrapper.

    Returns:
        int: number of modules wrapped.
    """
    wrapped = 0
    for name, module in model.named_modules():
        # Heuristic: module has q_proj/k_proj/v_proj/o_proj and num_heads attr
        if all(hasattr(module, attr) for attr in ("q_proj", "k_proj", "v_proj", "o_proj")) and hasattr(
            module, "num_heads"
        ):
            parent = _get_parent(model, name)
            if parent is None:
                continue
            parts = name.split(".")
            leaf = parts[-1]
            wrapper = SglKernelAttentionWrapper(module, backend)
            setattr(parent, leaf, wrapper)
            wrapped += 1
            logger.info("Patched attention module %s with SglKernelAttentionWrapper", name)
    if wrapped == 0:
        logger.warning("No attention modules patched; model structure may be unsupported")
    return wrapped


def _get_parent(root: nn.Module, qualname: str) -> Optional[nn.Module]:
    parts = qualname.split(".")
    if len(parts) == 1:
        return None
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p, None)
        if parent is None:
            return None
    return parent
