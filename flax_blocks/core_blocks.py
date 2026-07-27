"""Section 1 - Core neural-network blocks (Flax NNX, NHWC layout)."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Union

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from ._typecheck import typecheck


# ---------------------------------------------------------------------------
# Enums - canonical names for the string-keyed factories below
# ---------------------------------------------------------------------------

class _StrEnum(str, Enum):
    """Enum whose members *are* strings, so ``Norm.BATCH == "batch"``.

    Subclassing ``str`` keeps every existing string call site working while
    giving new code autocompletion and typo-safety.
    """

    def __str__(self) -> str:
        return self.value


class Activation(_StrEnum):
    """Activation functions available through :func:`get_activation`."""

    RELU = "relu"
    LEAKY_RELU = "leaky_relu"
    GELU = "gelu"
    SILU = "silu"
    SWISH = "swish"
    MISH = "mish"
    ELU = "elu"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    SOFTPLUS = "softplus"
    IDENTITY = "identity"

    def __call__(
        self, x: Float[Array, "..."]
    ) -> Float[Array, "..."]:
        """Apply the activation directly: ``Activation.GELU(x)``."""
        return get_activation(self)(x)


class Norm(_StrEnum):
    """Normalization layers available through :func:`build_norm`."""

    BATCH = "batch"
    LAYER = "layer"
    RMS = "rms"
    INSTANCE = "instance"
    GROUP = "group"
    NONE = "none"

    def __call__(self, num_features: int, *, rngs: nnx.Rngs) -> nnx.Module:
        """Build the layer directly: ``Norm.BATCH(64, rngs=rngs)``."""
        return build_norm(self, num_features, rngs=rngs)


class SkipMode(_StrEnum):
    """How :class:`SkipConnection` combines the branch with its input."""

    ADD = "add"
    CONCAT = "concat"


ActivationLike = Union[Activation, str]
NormLike = Union[Norm, str]
SkipModeLike = Union[SkipMode, str]


# ---------------------------------------------------------------------------
# Linear / Dense
# ---------------------------------------------------------------------------

Linear = nnx.Linear  # ``y = W x + b`` straight from the framework


# ---------------------------------------------------------------------------
# Convolutions  (NHWC layout - kernel = (kH, kW, Cin, Cout))
# ---------------------------------------------------------------------------

class ConvBlock(nnx.Module):
    """Conv -> Norm -> Activation: the standard conv "lego" piece."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "SAME",
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = True,
        norm: NormLike = Norm.LAYER,
        activation: ActivationLike = Activation.RELU,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        self.conv = nnx.Conv(
            in_ch, out_ch, (kernel_size, kernel_size),
            strides=strides, padding=padding,
            kernel_dilation=dilation, feature_group_count=groups,
            use_bias=use_bias, rngs=rngs,
        )
        self.norm = build_norm(norm, out_ch, rngs=rngs)
        self.act = get_activation(activation)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        return self.act(self.norm(self.conv(x)))


class DepthwiseSeparableConv(nnx.Module):
    """Depthwise 3x3 -> pointwise 1x1 (MobileNet-style)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 strides: int = 1, dilation: int = 1, use_bias: bool = True,
                 *, rngs: nnx.Rngs) -> None:
        self.depthwise = nnx.Conv(
            in_ch, in_ch, (kernel_size, kernel_size),
            strides=strides, kernel_dilation=dilation,
            feature_group_count=in_ch, use_bias=use_bias, rngs=rngs,
        )
        self.pointwise = nnx.Conv(in_ch, out_ch, (1, 1), use_bias=use_bias, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        return self.pointwise(self.depthwise(x))


class DilatedConv(nnx.Conv):
    """Atrous (dilated) convolution preserving spatial size."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 2, use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        super().__init__(in_ch, out_ch, (kernel_size, kernel_size),
                         padding="SAME", kernel_dilation=dilation,
                         use_bias=use_bias, rngs=rngs)


class GroupConv(nnx.Conv):
    """Grouped convolution - ancestor of depthwise conv."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 groups: int = 4, strides: int = 1, use_bias: bool = True,
                 *, rngs: nnx.Rngs) -> None:
        if in_ch % groups or out_ch % groups:
            raise ValueError("channels must divide groups")
        super().__init__(in_ch, out_ch, (kernel_size, kernel_size),
                         strides=strides, padding="SAME",
                         feature_group_count=groups, use_bias=use_bias, rngs=rngs)


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

def _mish(x: Float[Array, "..."]) -> Float[Array, "..."]:
    return x * jnp.tanh(jax.nn.softplus(x))


_ACTIVATIONS: dict[
    "Activation", Callable[[Float[Array, "..."]], Float[Array, "..."]]
] = {
    Activation.RELU: nnx.relu,
    Activation.LEAKY_RELU: lambda x: nnx.leaky_relu(x, 0.2),
    Activation.GELU: nnx.gelu,
    Activation.SILU: nnx.silu,
    Activation.SWISH: nnx.silu,
    Activation.MISH: _mish,
    Activation.ELU: nnx.elu,
    Activation.TANH: nnx.tanh,
    Activation.SIGMOID: nnx.sigmoid,
    Activation.SOFTPLUS: nnx.softplus,
    Activation.IDENTITY: lambda x: x,
}


def get_activation(
    name: ActivationLike,
) -> Callable[[Float[Array, "..."]], Float[Array, "..."]]:
    """Look up an activation function by :class:`Activation` member or name."""
    try:
        key = Activation(str(name).lower())
    except ValueError:
        raise KeyError(
            f"unknown activation '{name}', choose from "
            f"{[a.value for a in Activation]}"
        ) from None
    return _ACTIVATIONS[key]


# ---------------------------------------------------------------------------
# Normalizations
# ---------------------------------------------------------------------------

class InstanceNorm(nnx.Module):
    """Per-sample, per-channel normalization (Ulyanov et al. 2016).

    Implemented on top of :class:`nnx.GroupNorm` with one group per channel.
    """

    def __init__(self, num_features: int, *, epsilon: float = 1e-5,
                 rngs: nnx.Rngs) -> None:
        self.norm = nnx.GroupNorm(num_features, num_groups=num_features,
                                  epsilon=epsilon, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        return self.norm(x)


class WeightNorm(nnx.Module):
    """Salimans & Kingma 2016 - reparameterizes ``W = g * v / ||v||``.

    Wraps an existing :class:`nnx.Linear` (or any module whose ``kernel``
    parameter is a 2-D matrix) and replaces its kernel with a normalized
    version computed at every forward pass.
    """

    def __init__(self, base: nnx.Module) -> None:
        self.base = base
        kernel = base.kernel.value
        norm_axis = tuple(range(kernel.ndim - 1))
        self.g = nnx.Param(jnp.linalg.norm(
            kernel.reshape(-1, kernel.shape[-1]), axis=0))
        self.norm_axis = norm_axis

    @typecheck
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        v = self.base.kernel.value
        norm = jnp.sqrt(jnp.sum(v * v, axis=self.norm_axis, keepdims=True) + 1e-12)
        self.base.kernel.value = v * self.g.value / norm.squeeze()
        return self.base(x)


class SpectralNorm(nnx.Module):
    """Power-iteration spectral norm (Miyato et al. 2018).

    Wraps a 2-D :class:`nnx.Linear` and divides its kernel by the largest
    singular value approximated via one step of power iteration per call.
    """

    def __init__(self, base: nnx.Linear, *, rngs: nnx.Rngs) -> None:
        self.base = base
        out_features = base.kernel.value.shape[-1]
        self.u = nnx.Variable(
            jax.random.normal(rngs.params(), (out_features,)))

    @typecheck
    def __call__(
        self, x: Float[Array, "B D_in"]
    ) -> Float[Array, "B D_out"]:
        w = self.base.kernel.value
        u = self.u.value
        v = w @ u
        v = v / (jnp.linalg.norm(v) + 1e-12)
        u_new = w.T @ v
        u_new = u_new / (jnp.linalg.norm(u_new) + 1e-12)
        sigma = v @ w @ u_new
        self.u.value = jax.lax.stop_gradient(u_new)
        self.base.kernel.value = w / (sigma + 1e-12)
        return self.base(x)


class AdaIN(nnx.Module):
    """Adaptive Instance Normalization (Huang & Belongie 2017)."""

    def __init__(self, num_features: int, style_dim: int, *, rngs: nnx.Rngs) -> None:
        self.norm = InstanceNorm(num_features, rngs=rngs)
        self.fc = nnx.Linear(style_dim, num_features * 2, rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B H W C"],
        style: Float[Array, "B D_style"],
    ) -> Float[Array, "B H W C"]:
        gamma, beta = jnp.split(self.fc(style), 2, axis=-1)
        return (1 + gamma[:, None, None]) * self.norm(x) + beta[:, None, None]


class SPADE(nnx.Module):
    """Spatially-Adaptive (De)normalization (Park et al. 2019)."""

    def __init__(self, num_features: int, label_nc: int, hidden: int = 128,
                 *, rngs: nnx.Rngs) -> None:
        self.norm = nnx.BatchNorm(num_features, use_bias=False, use_scale=False,
                                  rngs=rngs)
        self.shared = nnx.Conv(label_nc, hidden, (3, 3), padding="SAME", rngs=rngs)
        self.gamma = nnx.Conv(hidden, num_features, (3, 3), padding="SAME", rngs=rngs)
        self.beta = nnx.Conv(hidden, num_features, (3, 3), padding="SAME", rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B H W C"],
        segmap: Float[Array, "B H_seg W_seg C_seg"],
    ) -> Float[Array, "B H W C"]:
        seg = jax.image.resize(
            segmap, x.shape[:-1] + (segmap.shape[-1],), method="nearest")
        actv = nnx.relu(self.shared(seg))
        return self.norm(x) * (1 + self.gamma(actv)) + self.beta(actv)


def build_norm(kind: NormLike, num_features: int, *,
               rngs: nnx.Rngs) -> nnx.Module:
    """Factory returning a 2-D normalization Module by :class:`Norm` or name."""
    try:
        kind = Norm(str(kind).lower())
    except ValueError:
        raise KeyError(
            f"unknown norm '{kind}', choose from {[n.value for n in Norm]}"
        ) from None
    if kind is Norm.BATCH:
        return nnx.BatchNorm(num_features, rngs=rngs)
    if kind is Norm.LAYER:
        return nnx.LayerNorm(num_features, rngs=rngs)
    if kind is Norm.RMS:
        return nnx.RMSNorm(num_features, rngs=rngs)
    if kind is Norm.INSTANCE:
        return InstanceNorm(num_features, rngs=rngs)
    if kind is Norm.GROUP:
        return nnx.GroupNorm(num_features,
                             num_groups=_gn_groups(num_features),
                             rngs=rngs)
    return _Identity()  # Norm.NONE


def _gn_groups(channels: int, target: int = 32) -> int:
    for g in range(min(channels, target), 0, -1):
        if channels % g == 0:
            return g
    return 1


class _Identity(nnx.Module):
    @typecheck
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return x


# ---------------------------------------------------------------------------
# Residual / Skip
# ---------------------------------------------------------------------------

class ResidualBlock(nnx.Module):
    """ResNet "basic block": ``y = act(F(x) + shortcut(x))``."""

    def __init__(self, in_ch: int, out_ch: int, strides: int = 1,
                 norm: NormLike = Norm.LAYER,
                 activation: ActivationLike = Activation.RELU,
                 *, rngs: nnx.Rngs) -> None:
        self.conv1 = ConvBlock(in_ch, out_ch, 3, strides=strides,
                               norm=norm, activation=activation, rngs=rngs)
        self.conv2 = ConvBlock(out_ch, out_ch, 3,
                               norm=norm, activation=Activation.IDENTITY, rngs=rngs)
        self.act = get_activation(activation)
        if strides != 1 or in_ch != out_ch:
            self.shortcut: nnx.Module = ConvBlock(in_ch, out_ch, 1, strides=strides,
                                                  norm=norm,
                                                  activation=Activation.IDENTITY,
                                                  rngs=rngs)
        else:
            self.shortcut = _Identity()

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        return self.act(self.conv2(self.conv1(x)) + self.shortcut(x))


class SkipConnection(nnx.Module):
    """Generic residual / concat wrapper: ``y = combine(f(x), x)``."""

    def __init__(self, fn: nnx.Module, mode: SkipModeLike = SkipMode.ADD) -> None:
        try:
            mode = SkipMode(str(mode).lower())
        except ValueError:
            raise ValueError("mode must be 'add' or 'concat'") from None
        self.fn = fn
        self.mode = mode

    @typecheck
    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        y = self.fn(x)
        return (x + y if self.mode is SkipMode.ADD
                else jnp.concatenate([x, y], axis=-1))
