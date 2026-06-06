"""Section 2 - Attention blocks.

Self / Multi-head / Cross / Causal / Sparse window / Linear / RoPE /
Relative position bias / Attention pooling.

All implementations route through :func:`F.scaled_dot_product_attention`
when possible, which transparently uses FlashAttention / mem-efficient
kernels on supported hardware.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# Scaled-dot-product helpers
# ---------------------------------------------------------------------------

@typecheck
def scaled_dot_product_attention(
    q: Float[Tensor, "B H Tq D"],
    k: Float[Tensor, "B H Tk D"],
    v: Float[Tensor, "B H Tk Dv"],
    mask: Optional[Shaped[Tensor, "..."]] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Float[Tensor, "B H Tq Dv"]:
    """``softmax(QK^T / sqrt(d)) V`` - delegating to fused kernels when available.

    Shapes: ``(B, H, Tq, D)`` for q/k and ``(B, H, Tk, Dv)`` for v. Query and key
    must share the head dim ``D`` (for the ``QK^T`` score); value may carry its own
    head dim ``Dv``, which the attention output then inherits.
    """
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal,
    )


# ---------------------------------------------------------------------------
# Multi-head attention (the workhorse)
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with optional causal masking.

    ``forward(query)``                  -> self-attention (Q = K = V = query),
    ``forward(query, key)``             -> Q from ``query``, K = V from ``key``,
    ``forward(query, key, value)``      -> general MHA with three sources.

    Query and key share ``dim`` (their projections feed the ``QK^T`` score, so
    they must land in the same space). The value stream may carry its own feature
    dim ``vdim``; ``v`` is kept in that space and the attention output therefore
    lives in ``vdim`` (``out_proj`` maps ``vdim -> vdim``). ``vdim`` defaults to
    ``dim``, recovering ordinary equal-width attention.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 bias: bool = True, causal: bool = False,
                 vdim: Optional[int] = None) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.vdim = vdim if vdim is not None else dim
        if self.vdim % num_heads != 0:
            raise ValueError("vdim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads             # head dim of q and k
        self.v_head_dim = self.vdim // num_heads     # head dim of v / output
        self.dim = dim
        self.causal = causal
        self.dropout = dropout

        # Separate q/k/v projections so the value stream may differ in feature
        # dim from query/key. q and k must land in `dim` (they form the QK^T
        # score); v stays in `vdim`, so the attention output is `vdim`.
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, self.vdim, bias=bias)
        self.out_proj = nn.Linear(self.vdim, self.vdim, bias=bias)

    @typecheck
    def forward(
        self,
        query: Float[Tensor, "B Tq D"],
        key: Optional[Float[Tensor, "B Tk D"]] = None,
        value: Optional[Float[Tensor, "B Tk D_val"]] = None,
        mask: Optional[Shaped[Tensor, "..."]] = None,
    ) -> Float[Tensor, "B Tq D_val"]:
        if key is None:
            key = query
        if value is None:
            value = key
        B, Tq, _ = query.shape
        H, D, Dv = self.num_heads, self.head_dim, self.v_head_dim

        q = self.q_proj(query).view(B, Tq,             H, D).transpose(1, 2)
        k = self.k_proj(key).view(B, key.shape[1],     H, D).transpose(1, 2)
        v = self.v_proj(value).view(B, value.shape[1], H, Dv).transpose(1, 2)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        out = out.transpose(1, 2).reshape(B, Tq, self.vdim)
        return self.out_proj(out)


class SelfAttention(MultiHeadAttention):
    """Single-head self-attention; thin alias of MHA with ``num_heads=1``."""

    def __init__(self, dim: int, dropout: float = 0.0, causal: bool = False):
        super().__init__(dim, num_heads=1, dropout=dropout, causal=causal)


class CausalSelfAttention(MultiHeadAttention):
    """Decoder-style attention that cannot peek at future tokens."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__(dim, num_heads, dropout, causal=True)


class CrossAttention(nn.Module):
    """Attention from a query stream to a separate key/value stream."""

    def __init__(self, dim: int, ctx_dim: Optional[int] = None,
                 num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        ctx_dim = ctx_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(ctx_dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B Tq D"],
        context: Float[Tensor, "B Tk D_ctx"],
        mask: Optional[Shaped[Tensor, "..."]] = None,
    ) -> Float[Tensor, "B Tq D"]:
        B, Tx, C = x.shape
        Tc = context.shape[1]
        q = self.q_proj(x).view(B, Tx, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(context).view(B, Tc, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)
        out = scaled_dot_product_attention(
            q, k, v, mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, Tx, C)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Sparse / Local attention
# ---------------------------------------------------------------------------

class WindowAttention(nn.Module):
    """Local attention restricted to fixed-size windows of length ``W``.

    Cost is ``O(T * W)`` rather than ``O(T^2)`` (Longformer-style).
    """

    def __init__(self, dim: int, num_heads: int = 8, window: int = 64,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.window = window
        self.attn = MultiHeadAttention(dim, num_heads, dropout)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        B, T, C = x.shape
        W = self.window
        pad = (W - T % W) % W
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        Tp = x.shape[1]
        x = x.view(B, Tp // W, W, C).reshape(B * (Tp // W), W, C)
        out = self.attn(x).view(B, Tp, C)
        return out[:, :T]


class LinearAttention(nn.Module):
    """Linear-complexity attention via the kernel trick (Performer-lite).

    Uses ``elu(x) + 1`` as positive feature map - simple, no random features.
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    @staticmethod
    def _phi(x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return F.elu(x) + 1.0

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self._phi(q), self._phi(k)
        kv = torch.einsum("bhnd,bhne->bhde", k, v)            # (B,H,D,D)
        z = 1.0 / (torch.einsum("bhnd,bhd->bhn", q, k.sum(2)) + 1e-6)
        out = torch.einsum("bhnd,bhde,bhn->bhne", q, kv, z)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class FlashAttention(MultiHeadAttention):
    """Just MHA - kept as a separate class for naming clarity.

    On CUDA with ``torch>=2.0`` :func:`F.scaled_dot_product_attention`
    automatically dispatches to the FlashAttention 2 kernel, so this is
    *exact* FlashAttention with no extra effort.
    """


# ---------------------------------------------------------------------------
# Positional information: RoPE & relative-position bias
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Rotary positional embedding (Su et al. 2021 / LLaMA)."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    @typecheck
    def forward(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> tuple[Float[Tensor, "T D"], Float[Tensor, "T D"]]:
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.outer(t, self.inv_freq.to(dtype=dtype))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


@typecheck
def apply_rotary(
    x: Float[Tensor, "B H T D"],
    cos: Float[Tensor, "T D"],
    sin: Float[Tensor, "T D"],
) -> Float[Tensor, "B H T D"]:
    """Rotate the last dim of ``x`` by RoPE angles. ``x`` is ``(B,H,T,D)``."""
    x1, x2 = x.chunk(2, dim=-1)
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


class RelativePositionBias(nn.Module):
    """T5-style logarithmic relative-position bias."""

    def __init__(self, num_heads: int, num_buckets: int = 32,
                 max_distance: int = 128, bidirectional: bool = True) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.bias = nn.Embedding(num_buckets, num_heads)

    def _bucket(
        self, rel_pos: Int[Tensor, "Tq Tk"]
    ) -> Int[Tensor, "Tq Tk"]:
        n = self.num_buckets
        ret = torch.zeros_like(rel_pos)
        if self.bidirectional:
            n //= 2
            ret += (rel_pos > 0).long() * n
            rel_pos = rel_pos.abs()
        else:
            rel_pos = (-rel_pos).clamp(min=0)
        max_exact = n // 2
        is_small = rel_pos < max_exact
        large = max_exact + (
            torch.log(rel_pos.float() / max_exact)
            / math.log(self.max_distance / max_exact) * (n - max_exact)
        ).long()
        large = large.clamp(max=n - 1)
        ret += torch.where(is_small, rel_pos, large)
        return ret

    @typecheck
    def forward(
        self, q_len: int, k_len: int, device: torch.device,
    ) -> Float[Tensor, "H Tq Tk"]:
        qpos = torch.arange(q_len, device=device)
        kpos = torch.arange(k_len, device=device)
        rel = kpos[None, :] - qpos[:, None]
        bucket = self._bucket(rel)
        return self.bias(bucket).permute(2, 0, 1)             # (H, q, k)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nn.Module):
    """Attention-weighted pooling with a single learnable query token."""

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = CrossAttention(dim, dim, num_heads)
        self.norm = nn.LayerNorm(dim)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B D"]:
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)
        return self.norm(self.attn(q, x)).squeeze(1)
