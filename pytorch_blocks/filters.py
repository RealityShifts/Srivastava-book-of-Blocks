"""Fixed (non-trainable) FIR filters for anti-aliasing (StyleGAN2/3-style).

The PyTorch twin of ``flax_blocks/filters.py``, op for op - the two produce
identical kernels, so a model ported between the frameworks keeps its
frequency response.

``Blur2d`` is the StyleGAN2 blur-after-upsample trick. ``AliasFreeActivation``
approximates StyleGAN3's alias-free nonlinearity (upsample -> nonlinearity ->
lowpass -> downsample) using a windowed-sinc FIR kernel, without the paper's
oversampling margins/cropping.

NCHW, like the rest of ``pytorch_blocks`` (the Flax originals are NHWC).
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from ._typecheck import typecheck


def _sinc_kernel_1d(taps: int, cutoff: float, fs: float = 1.0,
                    beta: float = 3.0) -> Tensor:
    """Windowed-sinc low-pass kernel (Kaiser window), unit DC gain.

    Written out rather than delegated to ``torch.kaiser_window`` so it stays a
    line-for-line match of the Flax implementation.
    """
    t = torch.arange(taps, dtype=torch.float32) - (taps - 1) / 2
    x = 2 * cutoff / fs * t
    sinc = torch.where(x == 0, torch.ones_like(x),
                       torch.sin(math.pi * x) / (math.pi * x))
    alpha = (taps - 1) / 2
    n = torch.arange(taps, dtype=torch.float32)
    r = (n - alpha) / alpha
    kaiser = torch.where(
        r.abs() <= 1,
        torch.special.i0(beta * torch.sqrt((1 - r ** 2).clamp(min=0)))
        / torch.special.i0(torch.tensor(beta)),
        torch.zeros_like(r),
    )
    k = sinc * kaiser
    return k / k.sum()


def sinc_kernel_2d(taps: int, cutoff: float, fs: float = 1.0) -> Tensor:
    """Separable 2-D windowed-sinc low-pass kernel."""
    k1 = _sinc_kernel_1d(taps, cutoff, fs)
    return torch.outer(k1, k1)


class Blur2d(nn.Module):
    """Fixed depthwise binomial blur (StyleGAN2 anti-alias filter).

    The kernel is a buffer, not a parameter: never trained, but it still has to
    follow the module across ``.to(device)`` and appear in ``state_dict`` - the
    same role ``nnx.Variable`` plays on the Flax side.
    """

    def __init__(self, channels: int,
                 kernel: Sequence[float] = (1.0, 3.0, 3.0, 1.0)) -> None:
        super().__init__()
        k1d = torch.tensor(kernel, dtype=torch.float32)
        k1d = k1d / k1d.sum()
        k2d = torch.outer(k1d, k1d)
        k2d = k2d / k2d.sum()
        # (C, 1, kh, kw) - one kernel per channel, applied with groups=C.
        self.register_buffer("kernel", k2d[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels

    @typecheck
    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        # padding="same" rather than an int: the default 4-tap kernel is even,
        # so it needs 1 before / 2 after, which no symmetric int pad expresses
        # and which is what JAX's "SAME" does.
        return F.conv2d(x, self.kernel.to(x.dtype), padding="same",
                        groups=self.channels)


class AliasFreeActivation(nn.Module):
    """2x upsample -> nonlinearity -> 2x downsample through one sinc FIR pair.

    Approximates StyleGAN3's alias-free nonlinearity. Deliberately omits the
    paper's oversampling margins and boundary cropping, so the cutoff is kept
    conservative rather than pushed near Nyquist.
    """

    def __init__(self, channels: int, cutoff_fraction: float = 0.2,
                 taps: int = 8,
                 act: Callable[[Tensor], Tensor] = F.leaky_relu) -> None:
        super().__init__()
        k = sinc_kernel_2d(taps, cutoff_fraction)
        # x4 on the upsampling kernel compensates the zero-stuffing: 3 of every
        # 4 cells feeding the filter are zeros, so DC gain would otherwise drop
        # to 1/4.
        self.register_buffer("up_kernel",
                             (k * 4.0)[None, None].repeat(channels, 1, 1, 1))
        self.register_buffer("down_kernel",
                             k[None, None].repeat(channels, 1, 1, 1))
        self.channels = channels
        self.taps = taps
        self.act = act

    @typecheck
    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        B, C, H, W = x.shape
        # Zero-stuff then depthwise-convolve, rather than a transposed conv:
        # matches the Flax version op for op (it had no grouped conv_transpose
        # available) and keeps the symmetric kernel exactly centred.
        stuffed = x.new_zeros((B, C, H * 2, W * 2))
        stuffed[:, :, ::2, ::2] = x
        up = F.conv2d(stuffed, self.up_kernel.to(x.dtype), padding="same",
                      groups=self.channels)
        up = self.act(up)
        # Stride 2 with an even kernel: JAX's "SAME" resolves to a symmetric
        # (taps - 2) / 2 pad here, which a plain int expresses exactly.
        return F.conv2d(up, self.down_kernel.to(x.dtype), stride=2,
                        padding=(self.taps - 2) // 2, groups=self.channels)
