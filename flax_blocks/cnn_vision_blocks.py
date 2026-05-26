"""Section 4 - CNN and vision blocks (NHWC).

Inception, DenseNet, Squeeze-and-Excitation, CBAM, Spatial Pyramid
Pooling, Feature Pyramid Network, ASPP, PixelShuffle, deformable conv /
attention.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from ._typecheck import typecheck

from .core_blocks import ConvBlock


# ---------------------------------------------------------------------------
# Inception
# ---------------------------------------------------------------------------

class InceptionBlock(nnx.Module):
    """Naive Inception-v1 module: 1x1, 3x3, 5x5 and pool branches."""

    def __init__(self, in_ch: int, c1: int = 64, c3: int = 96, c5: int = 16,
                 pool: int = 32, *, rngs: nnx.Rngs) -> None:
        self.b1 = ConvBlock(in_ch, c1, 1, rngs=rngs)
        self.b3a = ConvBlock(in_ch, c3, 1, rngs=rngs)
        self.b3b = ConvBlock(c3, c3, 3, rngs=rngs)
        self.b5a = ConvBlock(in_ch, c5, 1, rngs=rngs)
        self.b5b = ConvBlock(c5, c5, 5, rngs=rngs)
        self.bp = ConvBlock(in_ch, pool, 1, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        b1 = self.b1(x)
        b3 = self.b3b(self.b3a(x))
        b5 = self.b5b(self.b5a(x))
        bp = self.bp(nnx.max_pool(x, (3, 3), strides=(1, 1), padding="SAME"))
        return jnp.concatenate([b1, b3, b5, bp], axis=-1)


# ---------------------------------------------------------------------------
# DenseNet
# ---------------------------------------------------------------------------

class _DenseLayer(nnx.Module):
    def __init__(self, in_ch: int, growth: int, *, rngs: nnx.Rngs) -> None:
        self.bn1 = nnx.BatchNorm(in_ch, rngs=rngs)
        self.conv1 = nnx.Conv(in_ch, 4 * growth, (1, 1), use_bias=False, rngs=rngs)
        self.bn2 = nnx.BatchNorm(4 * growth, rngs=rngs)
        self.conv2 = nnx.Conv(4 * growth, growth, (3, 3),
                              padding="SAME", use_bias=False, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        h = self.conv1(nnx.relu(self.bn1(x)))
        h = self.conv2(nnx.relu(self.bn2(h)))
        return jnp.concatenate([x, h], axis=-1)


class DenseBlock(nnx.Module):
    """A dense block with ``num_layers`` densely-connected conv units."""

    def __init__(self, in_ch: int, num_layers: int = 4, growth: int = 32,
                 *, rngs: nnx.Rngs) -> None:
        layers = []
        c = in_ch
        for _ in range(num_layers):
            layers.append(_DenseLayer(c, growth, rngs=rngs))
            c += growth
        self.layers = layers
        self.out_channels = c

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Channel / spatial attention
# ---------------------------------------------------------------------------

class SqueezeExcitation(nnx.Module):
    """Hu et al. 2017 - SE channel attention."""

    def __init__(self, channels: int, ratio: int = 16,
                 *, rngs: nnx.Rngs) -> None:
        hidden = max(channels // ratio, 1)
        self.fc1 = nnx.Conv(channels, hidden, (1, 1), rngs=rngs)
        self.fc2 = nnx.Conv(hidden, channels, (1, 1), rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = nnx.sigmoid(self.fc2(nnx.relu(self.fc1(s))))
        return x * s


class CBAM(nnx.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, ratio: int = 16, kernel: int = 7,
                 *, rngs: nnx.Rngs) -> None:
        hidden = max(channels // ratio, 1)
        self.fc1 = nnx.Linear(channels, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, channels, rngs=rngs)
        self.spatial = nnx.Conv(2, 1, (kernel, kernel),
                                padding="SAME", rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        avg = jnp.mean(x, axis=(1, 2))
        mx = jnp.max(x, axis=(1, 2))
        ch = nnx.sigmoid(self.fc2(nnx.relu(self.fc1(avg))) +
                         self.fc2(nnx.relu(self.fc1(mx))))
        x = x * ch[:, None, None, :]
        sp = jnp.concatenate([jnp.mean(x, axis=-1, keepdims=True),
                              jnp.max(x, axis=-1, keepdims=True)], axis=-1)
        return x * nnx.sigmoid(self.spatial(sp))


# ---------------------------------------------------------------------------
# Multi-scale pooling / fusion
# ---------------------------------------------------------------------------

class SpatialPyramidPooling(nnx.Module):
    """Multi-scale pyramid average pooling, output is flattened."""

    def __init__(self, output_sizes: Sequence[int] = (1, 2, 4)) -> None:
        self.output_sizes = tuple(output_sizes)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B D_out"]:
        B, H, W, C = x.shape
        outs = []
        for s in self.output_sizes:
            kh = max(H // s, 1)
            kw = max(W // s, 1)
            pooled = nnx.avg_pool(x, (kh, kw), strides=(kh, kw))
            outs.append(pooled.reshape(B, -1))
        return jnp.concatenate(outs, axis=-1)


class FeaturePyramidNetwork(nnx.Module):
    """Top-down FPN with lateral 1x1 connections (Lin et al. 2017)."""

    def __init__(self, in_channels: Sequence[int], out_ch: int = 256,
                 *, rngs: nnx.Rngs) -> None:
        self.lat = [nnx.Conv(c, out_ch, (1, 1), rngs=rngs) for c in in_channels]
        self.smooth = [nnx.Conv(out_ch, out_ch, (3, 3),
                                padding="SAME", rngs=rngs) for _ in in_channels]

    @typecheck
    def __call__(
        self, feats: Sequence[Float[Array, "..."]]
    ) -> list[Float[Array, "..."]]:
        lats = [lat(f) for lat, f in zip(self.lat, feats)]
        outs = [lats[-1]]
        for i in range(len(lats) - 2, -1, -1):
            up = jax.image.resize(
                outs[-1],
                lats[i].shape[:1] + lats[i].shape[1:3] + (lats[-1].shape[-1],),
                method="nearest")
            outs.append(lats[i] + up)
        outs = outs[::-1]
        return [s(o) for s, o in zip(self.smooth, outs)]


class ASPP(nnx.Module):
    """Atrous Spatial Pyramid Pooling - DeepLab v3."""

    def __init__(self, in_ch: int, out_ch: int = 256,
                 dilations: Sequence[int] = (1, 6, 12, 18),
                 *, rngs: nnx.Rngs) -> None:
        self.branches: list[nnx.Module] = []
        for d in dilations:
            k = 1 if d == 1 else 3
            self.branches.append(ConvBlock(in_ch, out_ch, k, dilation=d, rngs=rngs))
        self.image_pool_conv = ConvBlock(in_ch, out_ch, 1, rngs=rngs)
        self.project = ConvBlock(out_ch * (len(dilations) + 1), out_ch, 1,
                                 rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        feats = [b(x) for b in self.branches]
        gp = jnp.mean(x, axis=(1, 2), keepdims=True)
        gp = self.image_pool_conv(gp)
        gp = jax.image.resize(gp, x.shape[:-1] + (gp.shape[-1],),
                              method="bilinear")
        return self.project(jnp.concatenate(feats + [gp], axis=-1))


# ---------------------------------------------------------------------------
# Pixel shuffle upsampler
# ---------------------------------------------------------------------------

@typecheck
def pixel_shuffle(
    x: Float[Array, "B H W C_packed"], scale: int
) -> Float[Array, "B H_out W_out C_out"]:
    """ESPCN sub-pixel rearrange. Input ``(B, H, W, C*scale^2)``."""
    B, H, W, C = x.shape
    if C % (scale * scale):
        raise ValueError("channels must be divisible by scale**2")
    out_c = C // (scale * scale)
    x = x.reshape(B, H, W, scale, scale, out_c)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    return x.reshape(B, H * scale, W * scale, out_c)


class PixelShuffleUpsample(nnx.Module):
    """Sub-pixel convolution: ESPCN / SRResNet upsampler."""

    def __init__(self, channels: int, scale: int = 2,
                 *, rngs: nnx.Rngs) -> None:
        self.scale = scale
        self.conv = nnx.Conv(channels, channels * scale * scale, (3, 3),
                             padding="SAME", rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H_out W_out C"]:
        return pixel_shuffle(self.conv(x), self.scale)


# ---------------------------------------------------------------------------
# Deformable conv & deformable attention (simplified)
# ---------------------------------------------------------------------------

def _bilinear_sample(
    image: Float[Array, "B H W C"],
    sample_y: Float[Array, "B *spatial"],
    sample_x: Float[Array, "B *spatial"],
) -> Float[Array, "B *spatial C"]:
    """Bilinear sample ``image`` (B, H, W, C) at given (y, x) coords.

    ``sample_y`` and ``sample_x`` may have any shape ``(B, *spatial)``.
    Returns ``(B, *spatial, C)``.
    """
    B, H, W, _ = image.shape
    y0 = jnp.floor(sample_y).astype(jnp.int32)
    x0 = jnp.floor(sample_x).astype(jnp.int32)
    y1, x1 = y0 + 1, x0 + 1
    wy1 = sample_y - y0
    wx1 = sample_x - x0
    wy0, wx0 = 1 - wy1, 1 - wx1
    y0c = jnp.clip(y0, 0, H - 1); y1c = jnp.clip(y1, 0, H - 1)
    x0c = jnp.clip(x0, 0, W - 1); x1c = jnp.clip(x1, 0, W - 1)
    b_shape = (B,) + (1,) * (sample_y.ndim - 1)
    b_idx = jnp.broadcast_to(jnp.arange(B).reshape(b_shape), sample_y.shape)
    Iaa = image[b_idx, y0c, x0c]
    Iab = image[b_idx, y0c, x1c]
    Iba = image[b_idx, y1c, x0c]
    Ibb = image[b_idx, y1c, x1c]
    return (Iaa * (wy0 * wx0)[..., None] + Iab * (wy0 * wx1)[..., None]
            + Iba * (wy1 * wx0)[..., None] + Ibb * (wy1 * wx1)[..., None])


class DeformableConv2d(nnx.Module):
    """Deformable convolution v1 (Dai et al. 2017), pure-JAX with bilinear sample."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 strides: int = 1, padding: int = 1,
                 *, rngs: nnx.Rngs) -> None:
        self.k = kernel_size
        self.strides = strides
        self.padding = padding
        self.offset = nnx.Conv(in_ch, 2 * kernel_size * kernel_size,
                               (kernel_size, kernel_size),
                               strides=strides, padding="SAME",
                               kernel_init=nnx.initializers.zeros,
                               bias_init=nnx.initializers.zeros, rngs=rngs)
        self.proj = nnx.Linear(in_ch * kernel_size * kernel_size, out_ch, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        B, H, W, C = x.shape
        k = self.k
        offset = self.offset(x)                                          # (B,Hout,Wout,2k^2)
        Hout, Wout = offset.shape[1:3]

        ys = jnp.arange(Hout, dtype=x.dtype)[:, None]
        xs = jnp.arange(Wout, dtype=x.dtype)[None, :]
        ky = jnp.arange(k, dtype=x.dtype)[:, None] - (k - 1) / 2
        kx = jnp.arange(k, dtype=x.dtype)[None, :] - (k - 1) / 2
        base_y = ys[..., None, None] * self.strides + ky[None, None] - self.padding
        base_x = xs[..., None, None] * self.strides + kx[None, None] - self.padding

        offs = offset.reshape(B, Hout, Wout, 2, k, k)
        sample_y = base_y[None] + offs[:, :, :, 0]
        sample_x = base_x[None] + offs[:, :, :, 1]

        sample_y = sample_y.reshape(B, Hout * k, Wout * k)
        sample_x = sample_x.reshape(B, Hout * k, Wout * k)
        sampled = _bilinear_sample(x, sample_y, sample_x)               # (B, Hout*k, Wout*k, C)
        sampled = sampled.reshape(B, Hout, k, Wout, k, C)
        sampled = jnp.transpose(sampled, (0, 1, 3, 2, 4, 5))
        sampled = sampled.reshape(B, Hout, Wout, k * k * C)
        return self.proj(sampled)


class DeformableAttention(nnx.Module):
    """Multi-head deformable attention (Zhu et al. - Deformable DETR).

    Each query produces ``num_points`` sampling offsets per head and pools
    values from those locations on a flat token grid ``(B, T=H*W, C)``.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_points: int = 4,
                 *, rngs: nnx.Rngs) -> None:
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads
        self.value = nnx.Linear(dim, dim, rngs=rngs)
        self.offset = nnx.Linear(dim, num_heads * num_points * 2,
                                 kernel_init=nnx.initializers.zeros,
                                 bias_init=nnx.initializers.zeros, rngs=rngs)
        self.weight = nnx.Linear(dim, num_heads * num_points, rngs=rngs)
        self.out = nnx.Linear(dim, dim, rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B T D"],
        hw: tuple[int, int],
    ) -> Float[Array, "B T D"]:
        B, T, _ = x.shape
        H, W = hw
        v = self.value(x).reshape(B, H, W, self.num_heads, self.head_dim)
        off = self.offset(x).reshape(B, T, self.num_heads, self.num_points, 2)
        w = jax.nn.softmax(self.weight(x).reshape(B, T, self.num_heads,
                                                  self.num_points), axis=-1)

        ys = jnp.arange(H, dtype=x.dtype)
        xs = jnp.arange(W, dtype=x.dtype)
        ref_y = jnp.broadcast_to(ys[:, None], (H, W)).reshape(T)
        ref_x = jnp.broadcast_to(xs[None, :], (H, W)).reshape(T)
        sample_y = ref_y[None, :, None, None] + off[..., 0]
        sample_x = ref_x[None, :, None, None] + off[..., 1]

        outs = []
        for h in range(self.num_heads):
            v_h = v[..., h, :]
            sy = sample_y[:, :, h].reshape(B, T * self.num_points)
            sx = sample_x[:, :, h].reshape(B, T * self.num_points)
            sampled = _bilinear_sample(v_h, sy, sx)                       # (B, T*P, D)
            sampled = sampled.reshape(B, T, self.num_points, self.head_dim)
            agg = (sampled * w[:, :, h, :, None]).sum(axis=2)            # (B, T, D)
            outs.append(agg)
        return self.out(jnp.concatenate(outs, axis=-1))
