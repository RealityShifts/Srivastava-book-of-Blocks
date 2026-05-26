"""Section 6 - GAN blocks.

Generator/discriminator templates, StyleGAN building blocks (equalized-LR
linear/conv, mapping network, style block, modulated conv), minibatch
standard-deviation, and a helper for progressive growing.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# Equalized-learning-rate primitives  (Karras et al. 2017)
# ---------------------------------------------------------------------------

class EqualLinear(nn.Module):
    """Equalized-LR linear used by Progressive GAN / StyleGAN.

    The kernel is sampled from ``N(0, 1)`` and scaled at runtime by
    ``gain / sqrt(fan_in)``; ``lr_mul`` further multiplies the effective
    learning rate (StyleGAN uses ``lr_mul=0.01`` for the mapping network).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 gain: float = 1.0, lr_mul: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) / lr_mul)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.scale = gain / math.sqrt(in_features) * lr_mul
        self.lr_mul = lr_mul

    @typecheck
    def forward(
        self, x: Float[Tensor, "*B D_in"]
    ) -> Float[Tensor, "*B D_out"]:
        bias = self.bias * self.lr_mul if self.bias is not None else None
        return F.linear(x, self.weight * self.scale, bias)


class EqualConv2d(nn.Module):
    """Equalized-LR 2-D convolution (StyleGAN family)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: Optional[int] = None,
                 bias: bool = True, gain: float = 1.0) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.stride = stride
        self.padding = padding
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = gain / math.sqrt(fan_in)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# Vanilla Gen / Disc blocks
# ---------------------------------------------------------------------------

class GeneratorBlock(nn.Module):
    """Up-sample -> conv -> norm -> activation."""

    def __init__(self, in_ch: int, out_ch: int, scale: int = 2) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, mode="nearest")
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_ch)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return F.relu(self.norm(self.conv(self.up(x))))


class DiscriminatorBlock(nn.Module):
    """Strided conv -> norm -> LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2,
                 use_norm: bool = True) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride, 1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=True) if use_norm else nn.Identity()

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return F.leaky_relu(self.norm(self.conv(x)), 0.2, inplace=True)


# ---------------------------------------------------------------------------
# Style components
# ---------------------------------------------------------------------------

class MappingNetwork(nn.Module):
    """StyleGAN ``z -> w`` mapping network (8-layer MLP)."""

    def __init__(self, z_dim: int = 512, w_dim: int = 512, num_layers: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers += [nn.Linear(z_dim if i == 0 else w_dim, w_dim),
                       nn.LeakyReLU(0.2, inplace=True)]
        self.net = nn.Sequential(*layers)

    @typecheck
    def forward(
        self, z: Float[Tensor, "B D_z"]
    ) -> Float[Tensor, "B D_w"]:
        z = z * z.pow(2).mean(dim=1, keepdim=True).add(1e-8).rsqrt()   # pixel-norm
        return self.net(z)


class StyleBlock(nn.Module):
    """Combines :class:`ModulatedConv2d` with noise injection and an activation."""

    def __init__(self, in_ch: int, out_ch: int, w_dim: int,
                 kernel_size: int = 3) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_ch, out_ch, kernel_size, w_dim)
        self.noise_strength = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B C_in H W"],
        w: Float[Tensor, "B D_w"],
    ) -> Float[Tensor, "B C_out H W"]:
        x = self.conv(x, w)
        noise = torch.randn(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        x = x + noise * self.noise_strength
        return F.leaky_relu(x + self.bias[None, :, None, None], 0.2, inplace=True)


class ModulatedConv2d(nn.Module):
    """StyleGAN2 modulated/demodulated convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, w_dim: int,
                 demodulate: bool = True) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.demodulate = demodulate
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size)
            / math.sqrt(in_ch * kernel_size * kernel_size))
        self.style = nn.Linear(w_dim, in_ch)
        nn.init.ones_(self.style.bias)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B C_in H W"],
        w: Float[Tensor, "B D_w"],
    ) -> Float[Tensor, "B C_out H W"]:
        B, C, H, W = x.shape
        s = self.style(w)                                                 # (B, in_ch)
        weight = self.weight[None] * s[:, None, :, None, None]            # modulate
        if self.demodulate:
            d = weight.pow(2).sum(dim=[2, 3, 4]).add(1e-8).rsqrt()
            weight = weight * d[:, :, None, None, None]
        weight = weight.view(B * weight.shape[1], C, *weight.shape[3:])
        x = x.view(1, B * C, H, W)
        out = F.conv2d(x, weight, padding=self.padding, groups=B)
        return out.view(B, -1, H, W)


# AdaIN re-exported for completeness from core_blocks
from .core_blocks import AdaIN  # noqa: E402  (placed late to avoid cycles)


# ---------------------------------------------------------------------------
# Minibatch / Progressive Growing
# ---------------------------------------------------------------------------

class MinibatchStdDev(nn.Module):
    """Karras et al. 2017 - appends per-batch std as an extra feature map."""

    def __init__(self, group_size: int = 4) -> None:
        super().__init__()
        self.group_size = group_size

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        B, C, H, W = x.shape
        g = min(self.group_size, B)
        y = x.view(g, -1, C, H, W)
        y = y - y.mean(dim=0, keepdim=True)
        y = y.pow(2).mean(0).add(1e-8).sqrt()
        y = y.mean(dim=[1, 2, 3], keepdim=True)
        y = y.repeat(g, 1, H, W).view(B, 1, H, W)
        return torch.cat([x, y], dim=1)


class ProgressiveGrowing(nn.Module):
    """Smooth fade between low- and high-resolution outputs by parameter ``alpha``."""

    def __init__(self, low_res_block: nn.Module, high_res_block: nn.Module) -> None:
        super().__init__()
        self.low = low_res_block
        self.high = high_res_block
        self.alpha = 0.0                                                    # 0 -> low, 1 -> high

    def set_alpha(self, alpha: float) -> None:
        self.alpha = max(0.0, min(1.0, alpha))

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        lo = F.interpolate(self.low(x), scale_factor=2, mode="nearest")
        hi = self.high(x)
        return (1 - self.alpha) * lo + self.alpha * hi
