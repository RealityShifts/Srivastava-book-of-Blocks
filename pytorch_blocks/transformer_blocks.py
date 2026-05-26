"""Section 3 - Transformer blocks.

Encoder/decoder layers, the FFN family (vanilla, SwiGLU, GEGLU,
gated MLP) and Mixture-of-Experts (top-k routing).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Shaped
from ._typecheck import typecheck
from torch import Tensor

from .attention_blocks import (
    MultiHeadAttention,
    CausalSelfAttention,
    CrossAttention,
)


# ---------------------------------------------------------------------------
# Feed-Forward variants
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Vanilla two-layer MLP used inside transformer blocks."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 dropout: float = 0.0, activation: str = "gelu") -> None:
        super().__init__()
        hidden = hidden or 4 * dim
        act = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), act(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    @typecheck
    def forward(
        self, x: Float[Tensor, "*B D"]
    ) -> Float[Tensor, "*B D"]:
        return self.net(x)


class SwiGLU(nn.Module):
    """SwiGLU FFN used in LLaMA / PaLM: ``Linear(SiLU(W1 x) * W3 x) -> W2``."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden or int(2 * dim * 4 / 3)               # keep params ~ FFN(4*dim)
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.drop = nn.Dropout(dropout)

    @typecheck
    def forward(
        self, x: Float[Tensor, "*B D"]
    ) -> Float[Tensor, "*B D"]:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class GEGLU(nn.Module):
    """GEGLU FFN: variant of GLU with a GELU gate (Shazeer 2020)."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        hidden = hidden or 4 * dim
        self.proj_in = nn.Linear(dim, 2 * hidden)
        self.proj_out = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    @typecheck
    def forward(
        self, x: Float[Tensor, "*B D"]
    ) -> Float[Tensor, "*B D"]:
        a, b = self.proj_in(x).chunk(2, dim=-1)
        return self.drop(self.proj_out(a * F.gelu(b)))


# ---------------------------------------------------------------------------
# Encoder & Decoder
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Pre-norm encoder block (BERT/ViT style)."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B T D"],
        mask: Optional[Shaped[Tensor, "..."]] = None,
    ) -> Float[Tensor, "B T D"]:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Pre-norm decoder block: causal self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 ctx_dim: Optional[int] = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CausalSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, ctx_dim, num_heads, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), dropout)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B T D"],
        context: Optional[Float[Tensor, "B Tk D_ctx"]] = None,
        mask: Optional[Shaped[Tensor, "..."]] = None,
    ) -> Float[Tensor, "B T D"]:
        x = x + self.self_attn(self.norm1(x), mask=mask)
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------

class MixtureOfExperts(nn.Module):
    """Token-level top-k MoE FFN (Shazeer / Switch / Mixtral).

    Each token is routed to its ``top_k`` highest-scoring experts and the
    outputs are weighted by softmax over the selected gate logits.
    """

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 hidden: Optional[int] = None) -> None:
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(dim, hidden) for _ in range(num_experts))

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        logits = self.gate(flat)                                # (N, E)
        weights, idx = logits.topk(self.top_k, dim=-1)
        weights = weights.softmax(dim=-1).to(flat.dtype)

        out = torch.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            mask = (idx == e)
            if not mask.any():
                continue
            tok_idx, k_idx = mask.nonzero(as_tuple=True)
            y = expert(flat[tok_idx])
            out.index_add_(0, tok_idx, y * weights[tok_idx, k_idx, None])
        return out.view(B, T, C)


class SwitchMoE(MixtureOfExperts):
    """Switch-Transformer style MoE: each token visits exactly one expert."""

    def __init__(self, dim: int, num_experts: int = 8, hidden: Optional[int] = None):
        super().__init__(dim, num_experts, top_k=1, hidden=hidden)
