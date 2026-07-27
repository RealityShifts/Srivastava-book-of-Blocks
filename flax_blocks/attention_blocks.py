"""Section 2 - Attention blocks.

Self / Multi-head / Cross / Causal / Sparse window / Linear / RoPE /
Relative-position bias / Attention pooling.

Custom MHA-from-scratch built on ``nnx.Linear`` + ``jnp.einsum`` so the
block surface is uniform; we also expose ``flax.nnx.dot_product_attention``
for users who need its fused kernels.
"""

from __future__ import annotations

import math
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike
from jaxtyping import Array, Float, Int, Shaped
from ._typecheck import typecheck


# ---------------------------------------------------------------------------
# Scaled-dot-product attention (functional)
# ---------------------------------------------------------------------------

@typecheck
def scaled_dot_product_attention(
    q: Float[Array, "B Tq H D"],
    k: Float[Array, "B Tk H D"],
    v: Float[Array, "B Tk H D"],
    mask: Optional[Shaped[Array, "..."]] = None,
    is_causal: bool = False,
) -> Float[Array, "B Tq H D"]:
    """``softmax(QK^T / sqrt(d)) V``.  q/k/v are ``(B, T, H, D)``.

    Routes through :func:`flax.nnx.dot_product_attention` which dispatches
    to fused JAX kernels when available.
    """
    if is_causal:
        T = q.shape[1]
        causal = nnx.make_causal_mask(jnp.ones((1, T)))
        mask = causal if mask is None else nnx.combine_masks(mask, causal)
    return nnx.dot_product_attention(q, k, v, mask=mask)


# ---------------------------------------------------------------------------
# Multi-head attention
# ---------------------------------------------------------------------------

class MultiHeadAttention(nnx.Module):
    """Pre-norm-friendly multi-head attention with optional causal masking.

    ``__call__(query)``              -> self-attention (Q = K = V = query),
    ``__call__(query, key)``         -> Q from ``query``, K = V from ``key``,
    ``__call__(query, key, value)``  -> general MHA with three sources.

    All three tensors must share ``dim``; key and value may differ from query
    in sequence length. Self-attention uses the fused qkv matmul; otherwise
    the same ``qkv`` kernel is split into Wq / Wk / Wv (mirrors PyTorch MHA).
    """

    def __init__(self, dim: int, num_heads: int = 8, use_bias: bool = True,
                 causal: bool = False, *, rngs: nnx.Rngs) -> None:
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=use_bias, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, use_bias=use_bias, rngs=rngs)

    def _project_qkv(
        self,
        query: jax.Array,
        key: jax.Array,
        value: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Apply Wq/Wk/Wv slices of the fused qkv linear; returns (B,T,H,D) each."""
        Wq, Wk, Wv = jnp.split(self.qkv.kernel.value, 3, axis=-1)
        if self.qkv.bias is not None:
            bq, bk, bv = jnp.split(self.qkv.bias.value, 3, axis=-1)
        else:
            bq = bk = bv = None
        B, Tq, _ = query.shape
        Tk = key.shape[1]
        H, D = self.num_heads, self.head_dim
        q = (query @ Wq + bq).reshape(B, Tq, H, D)
        k = (key @ Wk + bk).reshape(B, Tk, H, D)
        v = (value @ Wv + bv).reshape(B, value.shape[1], H, D)
        return q, k, v

    @typecheck
    def __call__(
        self,
        query: Float[Array, "B Tq D"],
        key: Optional[Float[Array, "B Tk D"]] = None,
        value: Optional[Float[Array, "B Tk D"]] = None,
        mask: Optional[Shaped[Array, "..."]] = None,
    ) -> Float[Array, "B Tq D"]:
        B, Tq, C = query.shape
        H, D = self.num_heads, self.head_dim

        if key is None and value is None:
            qkv = self.qkv(query).reshape(B, Tq, 3, H, D)
            q, k, v = jnp.moveaxis(qkv, 2, 0)                    # (B,T,H,D) each
        else:
            if key is None:
                key = query
            if value is None:
                value = key
            q, k, v = self._project_qkv(query, key, value)

        out = scaled_dot_product_attention(
            q, k, v, mask=mask, is_causal=self.causal,
        )
        out = out.reshape(B, Tq, C)
        return self.out_proj(out)


class SelfAttention(MultiHeadAttention):
    """Single-head self-attention."""

    def __init__(self, dim: int, *, rngs: nnx.Rngs, causal: bool = False):
        super().__init__(dim, num_heads=1, causal=causal, rngs=rngs)


class CausalSelfAttention(MultiHeadAttention):
    """Decoder-style attention that cannot peek at future tokens."""

    def __init__(self, dim: int, num_heads: int = 8, *, rngs: nnx.Rngs):
        super().__init__(dim, num_heads, causal=True, rngs=rngs)


class CrossAttention(nnx.Module):
    """Attention from a query stream to a separate key/value stream."""

    def __init__(self, dim: int, ctx_dim: Optional[int] = None,
                 num_heads: int = 8, *, rngs: nnx.Rngs) -> None:
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        ctx_dim = ctx_dim or dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.kv_proj = nnx.Linear(ctx_dim, 2 * dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B Tq D"],
        context: Float[Array, "B Tk D_ctx"],
        mask: Optional[Shaped[Array, "..."]] = None,
    ) -> Float[Array, "B Tq D"]:
        B, Tx, _ = x.shape
        Tc = context.shape[1]
        q = self.q_proj(x).reshape(B, Tx, self.num_heads, self.head_dim)
        kv = self.kv_proj(context).reshape(B, Tc, 2, self.num_heads, self.head_dim)
        k, v = jnp.moveaxis(kv, 2, 0)
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        return self.out_proj(out.reshape(B, Tx, -1))


# ---------------------------------------------------------------------------
# Sparse / Local / Linear attention
# ---------------------------------------------------------------------------

class WindowAttention(nnx.Module):
    """Local attention restricted to fixed-size windows."""

    def __init__(self, dim: int, num_heads: int = 8, window: int = 64,
                 *, rngs: nnx.Rngs) -> None:
        self.window = window
        self.attn = MultiHeadAttention(dim, num_heads, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        B, T, C = x.shape
        W = self.window
        pad = (W - T % W) % W
        if pad:
            x = jnp.pad(x, ((0, 0), (0, pad), (0, 0)))
        Tp = x.shape[1]
        x = x.reshape(B * (Tp // W), W, C)
        out = self.attn(x).reshape(B, Tp, C)
        return out[:, :T]


class LinearAttention(nnx.Module):
    """Linear-complexity attention via the kernel trick (Performer-lite)."""

    def __init__(self, dim: int, num_heads: int = 8,
                 *, rngs: nnx.Rngs) -> None:
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nnx.Linear(dim, 3 * dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(dim, dim, rngs=rngs)

    @staticmethod
    def _phi(x: Float[Array, "..."]) -> Float[Array, "..."]:
        return jax.nn.elu(x) + 1.0

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        q, k = self._phi(q), self._phi(k)
        kv = jnp.einsum("bnhd,bnhe->bhde", k, v)
        z = 1.0 / (jnp.einsum("bnhd,bhd->bnh", q, k.sum(axis=1)) + 1e-6)
        out = jnp.einsum("bnhd,bhde,bnh->bnhe", q, kv, z)
        return self.out_proj(out.reshape(B, T, -1))


class FlashAttention(MultiHeadAttention):
    """Just MHA - kept as a separate name for clarity.

    JAX's :func:`flax.nnx.dot_product_attention` already lowers to
    cuDNN / cuDNN-flash on supported backends.
    """


# ---------------------------------------------------------------------------
# Positional information
# ---------------------------------------------------------------------------

class RotaryEmbedding(nnx.Module):
    """Rotary positional embedding (Su et al. 2021 / LLaMA)."""

    def __init__(self, head_dim: int, base: float = 10_000.0) -> None:
        if head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        self.head_dim = head_dim
        self.base = base

    @typecheck
    def __call__(
        self, seq_len: int, dtype: DTypeLike = jnp.float32,
    ) -> tuple[Float[Array, "T D"], Float[Array, "T D"]]:
        inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.head_dim, 2,
                                                   dtype=dtype) / self.head_dim))
        t = jnp.arange(seq_len, dtype=dtype)
        freqs = jnp.outer(t, inv_freq)
        emb = jnp.concatenate([freqs, freqs], axis=-1)
        return jnp.cos(emb), jnp.sin(emb)


@typecheck
def apply_rotary(
    x: Float[Array, "... T D"],
    cos: Float[Array, "T D"],
    sin: Float[Array, "T D"],
) -> Float[Array, "... T D"]:
    """Rotate the last dim of ``x`` by RoPE angles. ``x`` is ``(..., T, D)``."""
    x1, x2 = jnp.split(x, 2, axis=-1)
    rotated = jnp.concatenate([-x2, x1], axis=-1)
    return x * cos + rotated * sin


class RelativePositionBias(nnx.Module):
    """T5-style logarithmic relative-position bias."""

    def __init__(self, num_heads: int, num_buckets: int = 32,
                 max_distance: int = 128, bidirectional: bool = True,
                 *, rngs: nnx.Rngs) -> None:
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        self.bias = nnx.Embed(num_buckets, num_heads, rngs=rngs)

    def _bucket(
        self, rel_pos: Int[Array, "Tq Tk"]
    ) -> Int[Array, "Tq Tk"]:
        n = self.num_buckets
        ret = jnp.zeros_like(rel_pos)
        if self.bidirectional:
            n = n // 2
            ret = ret + (rel_pos > 0).astype(rel_pos.dtype) * n
            rel_pos = jnp.abs(rel_pos)
        else:
            rel_pos = jnp.maximum(-rel_pos, 0)
        max_exact = n // 2
        is_small = rel_pos < max_exact
        large = max_exact + (
            jnp.log(rel_pos.astype(jnp.float32) / max_exact)
            / math.log(self.max_distance / max_exact) * (n - max_exact)
        ).astype(rel_pos.dtype)
        large = jnp.minimum(large, n - 1)
        return ret + jnp.where(is_small, rel_pos, large)

    @typecheck
    def __call__(
        self, q_len: int, k_len: int
    ) -> Float[Array, "H Tq Tk"]:
        qpos = jnp.arange(q_len)
        kpos = jnp.arange(k_len)
        rel = kpos[None, :] - qpos[:, None]
        bucket = self._bucket(rel)
        return jnp.transpose(self.bias(bucket), (2, 0, 1))   # (H, q, k)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

class AttentionPooling(nnx.Module):
    """Attention-weighted pooling with a single learnable query token."""

    def __init__(self, dim: int, num_heads: int = 8, *, rngs: nnx.Rngs) -> None:
        self.query = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, dim)) * 0.02)
        self.attn = CrossAttention(dim, dim, num_heads, rngs=rngs)
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B D"]:
        B = x.shape[0]
        q = jnp.broadcast_to(self.query.value, (B, 1, self.query.value.shape[-1]))
        return self.norm(self.attn(q, x))[:, 0]
