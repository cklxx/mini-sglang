"""Minimal Qwen-style stack using the lightweight sglang layers."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .sglang_attention import RadixAttention
from .sglang_layers import (
    LayerCommunicator,
    LogitsProcessor,
    ParallelLMHead,
    QKVParallelLinear,
    QwenMLP,
    RMSNorm,
    VocabParallelEmbedding,
)

try:  # pragma: no cover - optional
    from sglang.srt.layers.rotary_embedding import get_rope
except Exception:
    get_rope = None  # type: ignore


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.total_num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // num_heads
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            num_heads,
            total_num_kv_heads=num_kv_heads,
            bias=False,
        )
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.attn = RadixAttention(
            num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=self.head_dim, layer_id=0
        )
        self.rotary_emb = (
            get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=max_position_embeddings,
                base=10000,
            )
            if get_rope
            else None
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        total = self.total_num_heads * self.head_dim
        kv = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([total, kv, kv], dim=-1)
        q = q.view(
            hidden_states.shape[0], hidden_states.shape[1], self.total_num_heads, self.head_dim
        )
        k = k.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_kv_heads, self.head_dim
        )
        v = v.view(
            hidden_states.shape[0], hidden_states.shape[1], self.num_kv_heads, self.head_dim
        )
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_out = self.attn(q, k, v, save_kv_cache=True)
        attn_out = attn_out.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return self.o_proj(attn_out)


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.attention = Qwen3Attention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = QwenMLP(hidden_size, intermediate_size)
        self.layer_communicator = LayerCommunicator()

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_out = self.attention(positions, hidden_states)
        hidden_states = self.layer_communicator.postprocess_layer(attn_out, residual)[0]
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(hidden_states)
        hidden_states = self.layer_communicator.postprocess_layer(mlp_out, residual)[0]
        return hidden_states


class Qwen3Model(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    num_kv_heads=num_kv_heads,
                    rms_norm_eps=rms_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.lm_head = ParallelLMHead(vocab_size, hidden_size)
        self.logits_processor = LogitsProcessor(None)
        self.max_position_embeddings = max_position_embeddings

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        positions = (
            torch.arange(start_pos, start_pos + input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0)
        )
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.logits_processor(input_ids, hidden_states, self.lm_head)
        return logits
