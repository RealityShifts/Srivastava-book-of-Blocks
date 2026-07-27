"""Section 3 - Transformer blocks.

Encoder/decoder layers, FFN family (vanilla, SwiGLU, GEGLU) and a
top-k Mixture-of-Experts router.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Shaped
from ._typecheck import typecheck
from .core_blocks import Activation, ActivationLike, get_activation

from .attention_blocks import (
    MultiHeadAttention,
    CausalSelfAttention,
    CrossAttention,
)


# ---------------------------------------------------------------------------
# FFN variants
# ---------------------------------------------------------------------------

class FeedForward(nnx.Module):
    """Vanilla two-layer MLP used inside transformer blocks."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 activation: ActivationLike = Activation.GELU,
                 *, rngs: nnx.Rngs) -> None:
        hidden = hidden or 4 * dim
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, dim, rngs=rngs)
        self.act = get_activation(activation)

    @typecheck
    def __call__(
        self, x: Float[Array, "*B D"]
    ) -> Float[Array, "*B D"]:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLU(nnx.Module):
    """SwiGLU FFN: ``Linear(SiLU(W1 x) * W3 x) -> W2``  (LLaMA / PaLM)."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 *, rngs: nnx.Rngs) -> None:
        hidden = hidden or int(2 * dim * 4 / 3)
        self.w1 = nnx.Linear(dim, hidden, use_bias=False, rngs=rngs)
        self.w2 = nnx.Linear(hidden, dim, use_bias=False, rngs=rngs)
        self.w3 = nnx.Linear(dim, hidden, use_bias=False, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "*B D"]
    ) -> Float[Array, "*B D"]:
        return self.w2(nnx.silu(self.w1(x)) * self.w3(x))


class GEGLU(nnx.Module):
    """GEGLU FFN: GLU variant gated by GELU (Shazeer 2020)."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 *, rngs: nnx.Rngs) -> None:
        hidden = hidden or 4 * dim
        self.proj_in = nnx.Linear(dim, 2 * hidden, rngs=rngs)
        self.proj_out = nnx.Linear(hidden, dim, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "*B D"]
    ) -> Float[Array, "*B D"]:
        a, b = jnp.split(self.proj_in(x), 2, axis=-1)
        return self.proj_out(a * nnx.gelu(b))


# ---------------------------------------------------------------------------
# Encoder & Decoder
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nnx.Module):
    """Pre-norm encoder block (BERT/ViT style)."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 *, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.attn = MultiHeadAttention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B T D"],
        mask: Optional[Shaped[Array, "..."]] = None,
    ) -> Float[Array, "B T D"]:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerDecoderBlock(nnx.Module):
    """Pre-norm decoder block: causal self-attn + cross-attn + MLP."""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0,
                 ctx_dim: Optional[int] = None,
                 *, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.LayerNorm(dim, rngs=rngs)
        self.self_attn = CausalSelfAttention(dim, num_heads, rngs=rngs)
        self.norm2 = nnx.LayerNorm(dim, rngs=rngs)
        self.cross_attn = CrossAttention(dim, ctx_dim, num_heads, rngs=rngs)
        self.norm3 = nnx.LayerNorm(dim, rngs=rngs)
        self.mlp = FeedForward(dim, int(dim * mlp_ratio), rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B T D"],
        context: Optional[Float[Array, "B Tk D_ctx"]] = None,
        mask: Optional[Shaped[Array, "..."]] = None,
    ) -> Float[Array, "B T D"]:
        x = x + self.self_attn(self.norm1(x), mask=mask)
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Mixture of Experts
# ---------------------------------------------------------------------------

class MixtureOfExperts(nnx.Module):
    """Token-level top-k MoE FFN (Shazeer / Switch / Mixtral).

    Each token is routed to its ``top_k`` highest-scoring experts and the
    outputs are combined with the soft-maxed gate weights.
    """

    def __init__(self, dim: int, num_experts: int = 8, top_k: int = 2,
                 hidden: Optional[int] = None, *, rngs: nnx.Rngs) -> None:
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nnx.Linear(dim, num_experts, use_bias=False, rngs=rngs)
        self.experts = [FeedForward(dim, hidden, rngs=rngs)
                        for _ in range(num_experts)]

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        B, T, C = x.shape
        flat = x.reshape(-1, C)
        logits = self.gate(flat)
        top_vals, top_idx = jax.lax.top_k(logits, self.top_k)
        weights = jax.nn.softmax(top_vals, axis=-1)

        out = jnp.zeros_like(flat)
        for e, expert in enumerate(self.experts):
            mask = (top_idx == e)                                # (N, k)
            if not mask.any():
                continue
            tok_w = jnp.where(mask, weights, 0.0).sum(axis=-1)   # (N,)
            y = expert(flat) * tok_w[:, None]
            out = out + y
        return out.reshape(B, T, C)


class SwitchMoE(MixtureOfExperts):
    """Switch-Transformer style MoE: each token visits exactly one expert."""

    def __init__(self, dim: int, num_experts: int = 8,
                 hidden: Optional[int] = None, *, rngs: nnx.Rngs) -> None:
        super().__init__(dim, num_experts, top_k=1, hidden=hidden, rngs=rngs)
