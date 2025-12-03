"""Lightweight single-device reimplementations of sglang building blocks.

These mirror the class names and signatures of sglang components but avoid
distributed dependencies. They are intended for CPU/MPS development and to
provide a drop-in surface for CUDA/sgl_kernel backends when available.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


# ----------------------------
# Core layers (single-device)
# ----------------------------


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class LayerCommunicator:
    """Placeholder for sglang's LayerCommunicator; single device no-ops."""

    def pre_layer(self, hidden: torch.Tensor, residual: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return hidden, residual

    def postprocess_layer(self, hidden: torch.Tensor, residual: torch.Tensor, *_: object) -> Tuple[torch.Tensor, torch.Tensor]:
        return hidden + residual, residual


class VocabParallelEmbedding(nn.Module):
    """Single-device embedding compatible with sglang API."""

    def __init__(self, num_embeddings: int, embedding_dim: int, org_num_embeddings: Optional[int] = None, **_: object) -> None:
        super().__init__()
        self.org_num_embeddings = org_num_embeddings or num_embeddings
        self.embed = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids)

    @property
    def weight(self) -> torch.Tensor:  # type: ignore[override]
        return self.embed.weight


class ParallelLMHead(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, **_: object) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden_states)

    @property
    def weight(self) -> torch.Tensor:  # type: ignore[override]
        return self.proj.weight

    @weight.setter
    def weight(self, value: torch.Tensor) -> None:  # type: ignore[override]
        with torch.no_grad():
            self.proj.weight.copy_(value)


# ----------------------------
# Linear blocks
# ----------------------------


class QKVParallelLinear(nn.Module):
    """Single-device QKV projection; API-compatible stub."""

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.hidden_size = hidden_size
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size
        self.proj = nn.Linear(hidden_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.proj(x), None


class RowParallelLinear(nn.Module):
    """Single-device row-parallel stub (returns output, None)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False, **_: object) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return self.linear(x), None


class MergedColumnParallelLinear(nn.Module):
    """Produces two outputs from one projection (gate + up)."""

    def __init__(self, input_size: int, output_size: int, bias: bool = False, **_: object) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        out = self.linear(x)
        # Split equally for gate and up branches
        split = out.shape[-1] // 2
        gate = out[..., :split]
        up = out[..., split:]
        return torch.stack([gate, up], dim=0), None, None


# ----------------------------
# Activation and MLP
# ----------------------------


def silu_and_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


class QwenMLP(nn.Module):
    """Simple Qwen-style MLP using merged gate/up and down projection."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False) -> None:
        super().__init__()
        # gate + up combined
        self.gate_up = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up(x)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down(silu_and_mul(gate, up))


# ----------------------------
# Logits processor (minimal)
# ----------------------------


class LogitsProcessor(nn.Module):
    """Minimal logits processor placeholder."""

    def __init__(self, config: Optional[object] = None) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        *_: object,
    ) -> torch.Tensor:
        logits = lm_head(hidden_states)
        return logits
