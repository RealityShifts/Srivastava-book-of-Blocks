"""Section 6 - GAN blocks (NHWC).

Generator/discriminator templates, StyleGAN building blocks (equalized-LR
linear/conv, mapping network, style block, modulated conv), AdaIN
re-export, minibatch standard-deviation, progressive growing helper.
"""

from __future__ import annotations

import math
from typing import Union

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray

from .core_blocks import AdaIN  # noqa: F401  re-exported for completeness


# ---------------------------------------------------------------------------
# Equalized-learning-rate primitives  (Karras et al. 2017)
# ---------------------------------------------------------------------------

class EqualLinear(nnx.Module):
    """Equalized-LR linear used by Progressive GAN / StyleGAN.

    The kernel is sampled from ``N(0, 1)`` and scaled at runtime by
    ``gain / sqrt(fan_in)``; ``lr_mul`` further multiplies the effective
    learning rate (StyleGAN uses ``lr_mul=0.01`` for the mapping network).
    """

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool = True, gain: float = 1.0,
                 lr_mul: float = 1.0, *, rngs: nnx.Rngs) -> None:
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(), (in_features, out_features)) / lr_mul)
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None
        self.scale = gain / math.sqrt(in_features) * lr_mul
        self.lr_mul = lr_mul

    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_out"]:
        out = x @ (self.weight.value * self.scale)
        if self.bias is not None:
            out = out + self.bias.value * self.lr_mul
        return out


class EqualConv2d(nnx.Module):
    """Equalized-LR 2-D convolution (NHWC, StyleGAN family)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 strides: Union[int, tuple[int, int]] = 1,
                 padding: str = "SAME", use_bias: bool = True,
                 gain: float = 1.0, *, rngs: nnx.Rngs) -> None:
        self.weight = nnx.Param(jax.random.normal(
            rngs.params(), (kernel_size, kernel_size, in_ch, out_ch)))
        self.bias = nnx.Param(jnp.zeros((out_ch,))) if use_bias else None
        self.strides = ((strides, strides)
                        if isinstance(strides, int) else tuple(strides))
        self.padding = padding
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = gain / math.sqrt(fan_in)

    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        out = jax.lax.conv_general_dilated(
            x, self.weight.value * self.scale,
            window_strides=self.strides, padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"))
        if self.bias is not None:
            out = out + self.bias.value
        return out


# ---------------------------------------------------------------------------
# Vanilla generator / discriminator block templates
# ---------------------------------------------------------------------------

class GeneratorBlock(nnx.Module):
    """Up-sample -> conv -> norm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2,
                 *, rngs: nnx.Rngs) -> None:
        self.scale = scale
        self.conv = nnx.Conv(in_ch, out_ch, (3, 3), padding="SAME", rngs=rngs)
        self.norm = nnx.BatchNorm(out_ch, rngs=rngs)

    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        B, H, W, C = x.shape
        x = jax.image.resize(x, (B, H * self.scale, W * self.scale, C),
                             method="nearest")
        return nnx.relu(self.norm(self.conv(x)))


class DiscriminatorBlock(nnx.Module):
    """Strided 4x4 conv -> instance-norm -> LeakyReLU(0.2)."""

    def __init__(self, in_ch: int, out_ch: int, strides: int = 2,
                 use_norm: bool = True, *, rngs: nnx.Rngs) -> None:
        self.conv = nnx.Conv(in_ch, out_ch, (4, 4),
                             strides=strides, padding="SAME", rngs=rngs)
        self.norm = (nnx.GroupNorm(out_ch, num_groups=out_ch, rngs=rngs)
                     if use_norm else None)

    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        h = self.conv(x)
        if self.norm is not None:
            h = self.norm(h)
        return nnx.leaky_relu(h, 0.2)


# ---------------------------------------------------------------------------
# Style components
# ---------------------------------------------------------------------------

class MappingNetwork(nnx.Module):
    """StyleGAN ``z -> w`` mapping network (8-layer MLP)."""

    def __init__(self, z_dim: int = 512, w_dim: int = 512, num_layers: int = 8,
                 *, rngs: nnx.Rngs) -> None:
        self.layers = [nnx.Linear(z_dim if i == 0 else w_dim, w_dim, rngs=rngs)
                       for i in range(num_layers)]

    def __call__(
        self, z: Float[Array, "B D_z"]
    ) -> Float[Array, "B D_w"]:
        z = z * jax.lax.rsqrt(jnp.mean(z * z, axis=-1, keepdims=True) + 1e-8)
        for layer in self.layers:
            z = nnx.leaky_relu(layer(z), 0.2)
        return z


class ModulatedConv2d(nnx.Module):
    """StyleGAN2 modulated/demodulated convolution (NHWC)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, w_dim: int,
                 demodulate: bool = True, *, rngs: nnx.Rngs) -> None:
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.k = kernel_size
        self.demodulate = demodulate
        scale = 1.0 / math.sqrt(in_ch * kernel_size * kernel_size)
        self.weight = nnx.Param(
            jax.random.normal(rngs.params(),
                              (kernel_size, kernel_size, in_ch, out_ch)) * scale)
        self.style = nnx.Linear(w_dim, in_ch, rngs=rngs)
        self.style.bias.value = jnp.ones_like(self.style.bias.value)

    def __call__(
        self,
        x: Float[Array, "B H W C_in"],
        w: Float[Array, "B D_w"],
    ) -> Float[Array, "B H W C_out"]:
        B = x.shape[0]
        s = self.style(w)                                        # (B, in_ch)
        weight = self.weight.value[None] * s[:, None, None, :, None]
        if self.demodulate:
            d = jax.lax.rsqrt(jnp.sum(weight ** 2, axis=(1, 2, 3)) + 1e-8)
            weight = weight * d[:, None, None, None, :]
        outputs = []
        for b in range(B):
            outputs.append(jax.lax.conv_general_dilated(
                x[b:b + 1], weight[b],
                window_strides=(1, 1), padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC")))
        return jnp.concatenate(outputs, axis=0)


class StyleBlock(nnx.Module):
    """ModulatedConv + noise injection + leaky-ReLU."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int,
                 kernel_size: int = 3, *, rngs: nnx.Rngs) -> None:
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel_size, w_dim, rngs=rngs)
        self.noise_strength = nnx.Param(jnp.zeros(()))
        self.bias = nnx.Param(jnp.zeros((out_ch,)))

    def __call__(
        self,
        x: Float[Array, "B H W C_in"],
        w: Float[Array, "B D_w"],
        key: PRNGKeyArray,
    ) -> Float[Array, "B H W C_out"]:
        x = self.conv(x, w)
        noise = jax.random.normal(key, x.shape[:3] + (1,))
        x = x + noise * self.noise_strength.value
        return nnx.leaky_relu(x + self.bias.value, 0.2)


# ---------------------------------------------------------------------------
# Minibatch / progressive growing
# ---------------------------------------------------------------------------

class MinibatchStdDev(nnx.Module):
    """Karras et al. 2017 - appends per-batch std as an extra feature map."""

    def __init__(self, group_size: int = 4) -> None:
        self.group_size = group_size

    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C_out"]:
        B, H, W, C = x.shape
        g = min(self.group_size, B)
        y = x.reshape(g, -1, H, W, C)
        y = y - jnp.mean(y, axis=0, keepdims=True)
        y = jnp.sqrt(jnp.mean(y ** 2, axis=0) + 1e-8)
        y = jnp.mean(y, axis=(1, 2, 3), keepdims=True)
        y = jnp.broadcast_to(y, (g, H, W, 1))
        y = jnp.broadcast_to(y[:, None], (g, B // g, H, W, 1)).reshape(B, H, W, 1)
        return jnp.concatenate([x, y], axis=-1)


class ProgressiveGrowing(nnx.Module):
    """Smooth fade between low- and high-resolution outputs by ``alpha``."""

    def __init__(self, low_res_block: nnx.Module,
                 high_res_block: nnx.Module) -> None:
        self.low = low_res_block
        self.high = high_res_block
        self.alpha = 0.0

    def set_alpha(self, alpha: float) -> None:
        self.alpha = max(0.0, min(1.0, alpha))

    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        lo = self.low(x)
        B, H, W, C = lo.shape
        lo = jax.image.resize(lo, (B, H * 2, W * 2, C), method="nearest")
        hi = self.high(x)
        return (1 - self.alpha) * lo + self.alpha * hi
