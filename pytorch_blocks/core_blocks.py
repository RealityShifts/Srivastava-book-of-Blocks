"""Section 1 - Core neural-network blocks.

Linear, the Conv family (1D/2D/3D + depthwise/separable/dilated/group),
all common activations, every standard normalization (incl. RMSNorm,
WeightNorm, SpectralNorm, AdaIN, SPADE), residual blocks and a
typed skip-connection helper.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# Linear / Dense
# ---------------------------------------------------------------------------

class Linear(nn.Linear):
    """Plain dense layer ``y = W x + b``.

    Thin wrapper around :class:`torch.nn.Linear` that uses Kaiming-uniform
    init by default - sensible for most ReLU-family activations.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias=bias)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.zeros_(self.bias)


# ---------------------------------------------------------------------------
# Convolutions
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv2D -> Norm -> Activation. The most common conv "lego" piece."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm: str = "layer",
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride, padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        self.norm = _build_norm2d(norm, out_ch)
        self.act = get_activation(activation)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1 (MobileNet-style)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, bias: bool = True) -> None:
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, pad,
            dilation=dilation, groups=in_ch, bias=bias,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return self.pointwise(self.depthwise(x))


class DilatedConv2d(nn.Conv2d):
    """Atrous (dilated) convolution preserving spatial size."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 2, bias: bool = True) -> None:
        pad = dilation * (kernel_size - 1) // 2
        super().__init__(in_ch, out_ch, kernel_size, padding=pad,
                         dilation=dilation, bias=bias)


class GroupConv2d(nn.Conv2d):
    """Grouped convolution - the precursor of depthwise conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 groups: int = 4, stride: int = 1, bias: bool = True) -> None:
        assert in_ch % groups == 0 and out_ch % groups == 0, "channels must divide groups"
        super().__init__(in_ch, out_ch, kernel_size,
                         stride=stride, padding=kernel_size // 2,
                         groups=groups, bias=bias)


class Conv1d(nn.Conv1d):
    """Alias for ``nn.Conv1d`` - kept for symmetry with the API surface."""


class Conv3d(nn.Conv3d):
    """Alias for ``nn.Conv3d`` - kept for symmetry with the API surface."""


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class Mish(nn.Module):
    """Mish activation: ``x * tanh(softplus(x))``."""

    @typecheck
    def forward(  # noqa: D401
        self, x: Float[Tensor, "..."]
    ) -> Float[Tensor, "..."]:
        return x * torch.tanh(F.softplus(x))


_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "relu": lambda: nn.ReLU(inplace=True),
    "leaky_relu": lambda: nn.LeakyReLU(0.2, inplace=True),
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "mish": Mish,
    "elu": nn.ELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "identity": nn.Identity,
}


def get_activation(name: str) -> nn.Module:
    """Look up an activation module by short string name."""
    name = name.lower()
    if name not in _ACTIVATIONS:
        raise KeyError(f"unknown activation '{name}', choose from {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


# ---------------------------------------------------------------------------
# Normalizations
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (LLaMA / T5-v1.1)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @typecheck
    def forward(
        self, x: Float[Tensor, "... D"]
    ) -> Float[Tensor, "... D"]:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * rms)


class AdaIN(nn.Module):
    """Adaptive Instance Normalization (Huang & Belongie 2017).

    Re-normalizes each instance/channel with mean/std taken from a style code.
    """

    def __init__(self, num_features: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features * 2)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B C H W"],
        style: Float[Tensor, "B D_style"],
    ) -> Float[Tensor, "B C H W"]:
        h = self.fc(style)
        gamma, beta = h.chunk(2, dim=-1)
        return (1 + gamma[:, :, None, None]) * self.norm(x) + beta[:, :, None, None]


class SPADE(nn.Module):
    """Spatially-Adaptive (De)normalization (Park et al. 2019)."""

    def __init__(self, num_features: int, label_nc: int, hidden: int = 128) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_features, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(label_nc, hidden, 3, padding=1), nn.ReLU(inplace=True))
        self.gamma = nn.Conv2d(hidden, num_features, 3, padding=1)
        self.beta = nn.Conv2d(hidden, num_features, 3, padding=1)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B C H W"],
        segmap: Float[Tensor, "B C_seg H_seg W_seg"],
    ) -> Float[Tensor, "B C H W"]:
        seg = F.interpolate(segmap, size=x.shape[-2:], mode="nearest")
        actv = self.shared(seg)
        return self.norm(x) * (1 + self.gamma(actv)) + self.beta(actv)


def weight_norm(module: nn.Module, name: str = "weight", dim: int = 0) -> nn.Module:
    """Re-export of :func:`torch.nn.utils.weight_norm` for completeness."""
    return nn.utils.weight_norm(module, name=name, dim=dim)


def spectral_norm(module: nn.Module, name: str = "weight",
                  n_power_iterations: int = 1) -> nn.Module:
    """Re-export of :func:`torch.nn.utils.spectral_norm`."""
    return nn.utils.spectral_norm(module, name=name,
                                  n_power_iterations=n_power_iterations)


def _build_norm2d(kind: str, num_features: int) -> nn.Module:
    kind = kind.lower()
    if kind == "batch":
        return nn.BatchNorm2d(num_features)
    if kind == "layer":
        return nn.GroupNorm(1, num_features)
    if kind == "instance":
        return nn.InstanceNorm2d(num_features, affine=True)
    if kind == "group":
        for g in range(min(32, num_features), 0, -1):
            if num_features % g == 0:
                return nn.GroupNorm(g, num_features)
        return nn.GroupNorm(1, num_features)
    if kind == "none":
        return nn.Identity()
    raise KeyError(f"unknown 2D norm '{kind}'")


# ---------------------------------------------------------------------------
# Residual / Skip
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """ResNet "basic block": ``y = act(F(x) + shortcut(x))``."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 norm: str = "layer", activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_ch, out_ch, 3, stride, norm=norm, activation=activation)
        self.conv2 = ConvBlock(out_ch, out_ch, 3, 1, norm=norm, activation="identity")
        self.act = get_activation(activation)
        if stride != 1 or in_ch != out_ch:
            self.shortcut: nn.Module = ConvBlock(in_ch, out_ch, 1, stride,
                                                 norm=norm, activation="identity")
        else:
            self.shortcut = nn.Identity()

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class SkipConnection(nn.Module):
    """Generic residual / skip wrapper: ``y = combine(f(x), x)``."""

    def __init__(self, fn: nn.Module, mode: str = "add") -> None:
        super().__init__()
        if mode not in {"add", "concat"}:
            raise ValueError("mode must be 'add' or 'concat'")
        self.fn = fn
        self.mode = mode

    @typecheck
    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        y = self.fn(x)
        return x + y if self.mode == "add" else torch.cat([x, y], dim=1)
