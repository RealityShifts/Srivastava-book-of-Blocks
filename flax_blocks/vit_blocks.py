"""Section 7 - Vision Transformer blocks (NHWC).

Patch embedding, learnable CLS token, Swin window / shifted-window
attention, MAE / BEiT masked-image-modeling head.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray
from ._typecheck import typecheck


# ---------------------------------------------------------------------------
# Patch + CLS
# ---------------------------------------------------------------------------

class PatchEmbedding(nnx.Module):
    """Image -> sequence of patch tokens via a strided conv."""

    def __init__(self, in_ch: int = 3, patch_size: int = 16, dim: int = 768,
                 *, rngs: nnx.Rngs) -> None:
        self.patch_size = patch_size
        self.proj = nnx.Conv(in_ch, dim, (patch_size, patch_size),
                             strides=patch_size, padding="VALID", rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B N D"]:
        x = self.proj(x)                                       # (B, H/p, W/p, D)
        B, Hp, Wp, D = x.shape
        return x.reshape(B, Hp * Wp, D)


class CLSToken(nnx.Module):
    """Learnable [CLS] token prepended to each sequence."""

    def __init__(self, dim: int, *, rngs: nnx.Rngs) -> None:
        self.token = nnx.Param(
            jax.random.truncated_normal(rngs.params(), -2, 2, (1, 1, dim)) * 0.02)

    @typecheck
    def __call__(
        self, x: Float[Array, "B N D"]
    ) -> Float[Array, "B N_plus_1 D"]:
        cls = jnp.broadcast_to(self.token.value,
                               (x.shape[0], 1, self.token.value.shape[-1]))
        return jnp.concatenate([cls, x], axis=1)


# ---------------------------------------------------------------------------
# Swin window attention
# ---------------------------------------------------------------------------

def _window_partition(
    x: Float[Array, "B H W C"], w: int
) -> Float[Array, "BnW WW C"]:
    """``(B, H, W, C) -> (B*nW, w*w, C)``."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // w, w, W // w, w, C)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(-1, w * w, C)


def _window_reverse(
    windows: Float[Array, "BnW WW C"], w: int, H: int, W: int
) -> Float[Array, "B H W C"]:
    B = windows.shape[0] // (H * W // (w * w))
    x = windows.reshape(B, H // w, W // w, w, w, -1)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(B, H, W, -1)


class SwinWindowAttention(nnx.Module):
    """Window attention with learned relative-position bias (used inside Swin)."""

    def __init__(self, dim: int, num_heads: int, window: int = 7,
                 *, rngs: nnx.Rngs) -> None:
        self.dim = dim
        self.window = window
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nnx.Linear(dim, 3 * dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

        self.rel_bias = nnx.Param(
            jax.random.truncated_normal(rngs.params(), -2, 2,
                                        ((2 * window - 1) ** 2, num_heads)) * 0.02)
        coords = jnp.stack(jnp.meshgrid(
            jnp.arange(window), jnp.arange(window), indexing="ij")).reshape(2, -1)
        rel = coords[:, :, None] - coords[:, None, :]
        rel = jnp.transpose(rel, (1, 2, 0))
        rel = rel.at[..., 0].add(window - 1).at[..., 1].add(window - 1)
        rel = rel.at[..., 0].multiply(2 * window - 1)
        self.rel_index = rel.sum(-1)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B N D"],
        mask: Optional[Float[Array, "..."]] = None,
    ) -> Float[Array, "B N D"]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = jnp.moveaxis(qkv, 2, 0)
        attn = jnp.einsum("bnhd,bmhd->bhnm", q, k) * (self.head_dim ** -0.5)

        bias = self.rel_bias.value[self.rel_index.reshape(-1)].reshape(N, N, -1)
        attn = attn + jnp.transpose(bias, (2, 0, 1))[None]

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask[None, :, None]
            attn = attn.reshape(-1, self.num_heads, N, N)
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhnm,bmhd->bnhd", attn, v).reshape(B, N, C)
        return self.proj(out)


class ShiftedWindowAttention(nnx.Module):
    """Cyclic-shift variant enabling cross-window communication (Swin)."""

    def __init__(self, dim: int, input_resolution: tuple[int, int],
                 num_heads: int, window: int = 7, shift: Optional[int] = None,
                 *, rngs: nnx.Rngs) -> None:
        self.input_resolution = input_resolution
        self.window = window
        self.shift = shift if shift is not None else window // 2
        self.attn = SwinWindowAttention(dim, num_heads, window, rngs=rngs)

        if self.shift > 0:
            H, W = input_resolution
            img_mask = jnp.zeros((1, H, W, 1))
            cnt = 0
            for h_slice in (slice(0, -window), slice(-window, -self.shift),
                            slice(-self.shift, None)):
                for w_slice in (slice(0, -window), slice(-window, -self.shift),
                                slice(-self.shift, None)):
                    img_mask = img_mask.at[:, h_slice, w_slice, :].set(cnt)
                    cnt += 1
            mask_windows = _window_partition(img_mask, window).squeeze(-1)
            attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
            attn_mask = jnp.where(attn_mask != 0, -100.0, 0.0)
            self.attn_mask = attn_mask
        else:
            self.attn_mask = None

    @typecheck
    def __call__(
        self, x: Float[Array, "B N D"]
    ) -> Float[Array, "B N D"]:
        B, N, C = x.shape
        H, W = self.input_resolution
        x = x.reshape(B, H, W, C)
        if self.shift > 0:
            x = jnp.roll(x, shift=(-self.shift, -self.shift), axis=(1, 2))
        windows = _window_partition(x, self.window)
        attn = self.attn(windows, mask=self.attn_mask)
        x = _window_reverse(attn, self.window, H, W)
        if self.shift > 0:
            x = jnp.roll(x, shift=(self.shift, self.shift), axis=(1, 2))
        return x.reshape(B, N, C)


# ---------------------------------------------------------------------------
# Masked image modeling
# ---------------------------------------------------------------------------

class MaskedImageModeling(nnx.Module):
    """MAE / BEiT-style head: mask a fraction of tokens and reconstruct them."""

    def __init__(self, dim: int, patch_size: int = 16, in_ch: int = 3,
                 mask_ratio: float = 0.75, *, rngs: nnx.Rngs) -> None:
        self.mask_ratio = mask_ratio
        self.mask_token = nnx.Param(
            jax.random.truncated_normal(rngs.params(), -2, 2, (1, 1, dim)) * 0.02)
        self.decoder = nnx.Linear(dim, patch_size * patch_size * in_ch, rngs=rngs)

    def random_masking(
        self,
        x: Float[Array, "B N D"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "B N_keep D"],
               Float[Array, "B N"],
               Float[Array, "B N"]]:
        """Returns ``(visible_tokens, restore_indices, mask)``."""
        B, N, _ = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = jax.random.uniform(key, (B, N))
        ids_shuffle = jnp.argsort(noise, axis=1)
        ids_restore = jnp.argsort(ids_shuffle, axis=1)
        ids_keep = ids_shuffle[:, :keep]
        x_visible = jnp.take_along_axis(x, ids_keep[:, :, None], axis=1)
        mask = jnp.concatenate(
            [jnp.zeros((B, keep)), jnp.ones((B, N - keep))], axis=1)
        mask = jnp.take_along_axis(mask, ids_restore, axis=1)
        return x_visible, ids_restore, mask

    def reconstruct(
        self,
        encoded: Float[Array, "B N_keep D"],
        ids_restore: Float[Array, "B N"],
    ) -> Float[Array, "B N P_sq_C"]:
        B, N = ids_restore.shape
        n_mask = N - encoded.shape[1]
        tokens = jnp.concatenate(
            [encoded,
             jnp.broadcast_to(self.mask_token.value,
                              (B, n_mask, self.mask_token.value.shape[-1]))],
            axis=1)
        tokens = jnp.take_along_axis(tokens, ids_restore[:, :, None], axis=1)
        return self.decoder(tokens)
