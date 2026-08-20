"""Fixed (non-trainable) FIR filters for anti-aliasing (StyleGAN2/3-style).

``Blur2d`` is the StyleGAN2 blur-after-upsample trick. ``AliasFreeActivation``
approximates StyleGAN3's alias-free nonlinearity (upsample -> nonlinearity ->
lowpass -> downsample) using a windowed-sinc FIR kernel, without the paper's
oversampling margins/cropping - see ``model/synthesis.py`` for where and why.
"""

from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from ._typecheck import typecheck


def _sinc_kernel_1d(taps: int, cutoff: float, fs: float = 1.0,
                     beta: float = 3.0) -> jnp.ndarray:
    """Windowed-sinc low-pass kernel (Kaiser window), unit DC gain."""
    t = jnp.arange(taps) - (taps - 1) / 2
    x = 2 * cutoff / fs * t
    sinc = jnp.where(x == 0, 1.0, jnp.sin(jnp.pi * x) / (jnp.pi * x))
    alpha = (taps - 1) / 2
    n = jnp.arange(taps)
    r = (n - alpha) / alpha
    kaiser = jnp.where(
        jnp.abs(r) <= 1,
        jax.scipy.special.i0(beta * jnp.sqrt(1 - r ** 2)) / jax.scipy.special.i0(beta),
        0.0,
    )
    k = sinc * kaiser
    return k / jnp.sum(k)


def sinc_kernel_2d(taps: int, cutoff: float, fs: float = 1.0) -> jnp.ndarray:
    """Separable 2-D windowed-sinc low-pass kernel."""
    k1 = _sinc_kernel_1d(taps, cutoff, fs)
    return jnp.outer(k1, k1)


class Blur2d(nnx.Module):
    """Fixed depthwise binomial blur (StyleGAN2 anti-alias filter)."""

    def __init__(self, channels: int,
                 kernel: Sequence[float] = (1.0, 3.0, 3.0, 1.0)) -> None:
        k1d = jnp.asarray(kernel, dtype=jnp.float32)
        k1d = k1d / jnp.sum(k1d)
        k2d = jnp.outer(k1d, k1d)
        k2d = k2d / jnp.sum(k2d)
        kernel_hwio = jnp.tile(k2d[:, :, None, None], (1, 1, 1, channels))
        self.kernel = nnx.Variable(kernel_hwio)
        self.channels = channels

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        return jax.lax.conv_general_dilated(
            x, self.kernel.value, window_strides=(1, 1), padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.channels)


class AliasFreeActivation(nnx.Module):
    """2x upsample -> nonlinearity -> 2x downsample, both via the same

    windowed-sinc FIR kernel. Approximates StyleGAN3's alias-free
    nonlinearity; deliberately omits the paper's oversampling margins and
    boundary cropping (see model/synthesis.py for the tradeoff), so the
    cutoff is kept conservative rather than pushed near Nyquist.
    """

    def __init__(self, channels: int, cutoff_fraction: float = 0.2,
                 taps: int = 8, act: Callable = nnx.leaky_relu) -> None:
        k = sinc_kernel_2d(taps, cutoff_fraction)
        self.up_kernel = nnx.Variable((k * 4.0).reshape(taps, taps, 1, 1))
        self.down_kernel = nnx.Variable(k.reshape(taps, taps, 1, 1))
        self.channels = channels
        self.act = act

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        # jax.lax.conv_transpose has no feature_group_count, so upsampling
        # is done as zero-stuffing + a depthwise conv with the (symmetric)
        # sinc kernel - equivalent to a depthwise transposed convolution.
        B, H, W, C = x.shape
        x_stuffed = jnp.zeros((B, H * 2, W * 2, C), dtype=x.dtype)
        x_stuffed = x_stuffed.at[:, ::2, ::2, :].set(x)
        up_kernel = jnp.tile(self.up_kernel.value, (1, 1, 1, self.channels))
        x_up = jax.lax.conv_general_dilated(
            x_stuffed, up_kernel, window_strides=(1, 1), padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.channels)
        x_up = self.act(x_up)
        down_kernel = jnp.tile(self.down_kernel.value, (1, 1, 1, self.channels))
        return jax.lax.conv_general_dilated(
            x_up, down_kernel, window_strides=(2, 2), padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.channels)
