"""Section 5 - UNet & diffusion blocks.

UNet (encoder-decoder + skips), down/up-sample blocks, sinusoidal time
embedding, classifier-free guidance helper, ControlNet hint encoder,
LoRA adapters, hypernetwork, IP-Adapter cross-attention.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int
from ._typecheck import typecheck

from .attention_blocks import CrossAttention


def _gn_groups(channels: int, target: int = 32) -> int:
    for g in range(min(channels, target), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nnx.Module):
    """Diffusion-style sinusoidal embedding of integer timesteps."""

    def __init__(self, dim: int) -> None:
        self.dim = dim

    @typecheck
    def __call__(
        self, t: Int[Array, "B"]
    ) -> Float[Array, "B D"]:
        half = self.dim // 2
        freqs = jnp.exp(
            -math.log(10_000) * jnp.arange(half) / max(half - 1, 1))
        args = t.astype(jnp.float32)[:, None] * freqs[None]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        if self.dim % 2:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb


class TimestepMLP(nnx.Module):
    """Sinusoidal embedding followed by a 2-layer MLP - used by DDPM/SD."""

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 *, rngs: nnx.Rngs) -> None:
        hidden = hidden or 4 * dim
        self.embed = SinusoidalTimeEmbedding(dim)
        self.fc1 = nnx.Linear(dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, hidden, rngs=rngs)

    @typecheck
    def __call__(
        self, t: Int[Array, "B"]
    ) -> Float[Array, "B D_hidden"]:
        return self.fc2(nnx.silu(self.fc1(self.embed(t))))


# ---------------------------------------------------------------------------
# Down / up-sampling
# ---------------------------------------------------------------------------

class DownsampleBlock(nnx.Module):
    """Strided 3x3 conv (or avg-pool) for halving spatial size."""

    def __init__(self, channels: int, mode: str = "conv",
                 *, rngs: nnx.Rngs) -> None:
        self.mode = mode
        if mode == "conv":
            self.op = nnx.Conv(channels, channels, (3, 3),
                               strides=2, padding="SAME", rngs=rngs)
        elif mode != "avg":
            raise ValueError(mode)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H_out W_out C"]:
        if self.mode == "avg":
            return nnx.avg_pool(x, (2, 2), strides=(2, 2))
        return self.op(x)


class UpsampleBlock(nnx.Module):
    """Restores spatial resolution via interp / transposed conv / pixel shuffle."""

    def __init__(self, channels: int, mode: str = "interp",
                 *, rngs: nnx.Rngs) -> None:
        self.mode = mode
        self.channels = channels
        if mode == "interp":
            self.conv = nnx.Conv(channels, channels, (3, 3),
                                 padding="SAME", rngs=rngs)
        elif mode == "transpose":
            self.op = nnx.ConvTranspose(channels, channels, (4, 4),
                                        strides=2, padding="SAME", rngs=rngs)
        elif mode == "shuffle":
            self.conv = nnx.Conv(channels, channels * 4, (3, 3),
                                 padding="SAME", rngs=rngs)
        elif mode == "blur":
            self.conv = nnx.Conv(channels, channels, (3, 3),
                                 padding="SAME", rngs=rngs)
            from .filters import Blur2d
            self.blur = Blur2d(channels)
        else:
            raise ValueError(mode)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H_out W_out C"]:
        if self.mode == "interp":
            B, H, W, C = x.shape
            x = jax.image.resize(x, (B, H * 2, W * 2, C), method="nearest")
            return self.conv(x)
        if self.mode == "blur":
            B, H, W, C = x.shape
            x = jax.image.resize(x, (B, H * 2, W * 2, C), method="nearest")
            x = self.blur(x)
            return self.conv(x)
        if self.mode == "transpose":
            return self.op(x)
        from .cnn_vision_blocks import pixel_shuffle
        return pixel_shuffle(self.conv(x), 2)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNetResBlock(nnx.Module):
    """Diffusion-style residual block conditioned on a time embedding."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 *, rngs: nnx.Rngs) -> None:
        self.norm1 = nnx.GroupNorm(in_ch, num_groups=_gn_groups(in_ch), rngs=rngs)
        self.conv1 = nnx.Conv(in_ch, out_ch, (3, 3), padding="SAME", rngs=rngs)
        self.t_proj = nnx.Linear(t_dim, out_ch, rngs=rngs)
        self.norm2 = nnx.GroupNorm(out_ch, num_groups=_gn_groups(out_ch), rngs=rngs)
        self.conv2 = nnx.Conv(out_ch, out_ch, (3, 3), padding="SAME", rngs=rngs)
        self.skip = (nnx.Conv(in_ch, out_ch, (1, 1), rngs=rngs)
                     if in_ch != out_ch else None)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B H W C_in"],
        t_emb: Float[Array, "B D_t"],
    ) -> Float[Array, "B H W C_out"]:
        h = self.conv1(nnx.silu(self.norm1(x)))
        h = h + self.t_proj(nnx.silu(t_emb))[:, None, None]
        h = self.conv2(nnx.silu(self.norm2(h)))
        return h + (self.skip(x) if self.skip is not None else x)


class UNet(nnx.Module):
    """A compact diffusion-style UNet operating on NHWC images."""

    def __init__(self, in_ch: int = 4, out_ch: int = 4, base: int = 64,
                 ch_mults: Sequence[int] = (1, 2, 4, 4), t_dim: int = 256,
                 *, rngs: nnx.Rngs) -> None:
        self.t_embed = TimestepMLP(t_dim, rngs=rngs)
        self.stem = nnx.Conv(in_ch, base, (3, 3), padding="SAME", rngs=rngs)

        downs = []
        chans = [base]
        c = base
        for i, m in enumerate(ch_mults):
            o = base * m
            r1 = UNetResBlock(c, o, 4 * t_dim, rngs=rngs)
            r2 = UNetResBlock(o, o, 4 * t_dim, rngs=rngs)
            down = DownsampleBlock(o, rngs=rngs) if i < len(ch_mults) - 1 else None
            downs.append((r1, r2, down))
            chans.extend([o, o])
            c = o
        self.downs = downs

        self.mid1 = UNetResBlock(c, c, 4 * t_dim, rngs=rngs)
        self.mid2 = UNetResBlock(c, c, 4 * t_dim, rngs=rngs)

        ups = []
        for i, m in enumerate(reversed(ch_mults)):
            o = base * m
            skip_c = chans.pop()
            skip_c2 = chans.pop()
            r1 = UNetResBlock(c + skip_c, o, 4 * t_dim, rngs=rngs)
            r2 = UNetResBlock(o + skip_c2, o, 4 * t_dim, rngs=rngs)
            up = UpsampleBlock(o, rngs=rngs) if i < len(ch_mults) - 1 else None
            ups.append((r1, r2, up))
            c = o
        self.ups = ups

        self.out_norm = nnx.GroupNorm(c, num_groups=_gn_groups(c), rngs=rngs)
        self.out_conv = nnx.Conv(c, out_ch, (3, 3), padding="SAME", rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B H W C_in"],
        t: Int[Array, "B"],
    ) -> Float[Array, "B H W C_out"]:
        t_emb = self.t_embed(t)
        h = self.stem(x)
        skips = [h]
        for r1, r2, down in self.downs:
            h = r1(h, t_emb); skips.append(h)
            h = r2(h, t_emb); skips.append(h)
            if down is not None:
                h = down(h)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        for r1, r2, up in self.ups:
            h = r1(jnp.concatenate([h, skips.pop()], axis=-1), t_emb)
            h = r2(jnp.concatenate([h, skips.pop()], axis=-1), t_emb)
            if up is not None:
                h = up(h)
        return self.out_conv(nnx.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Noise prediction wrapper + classifier-free guidance
# ---------------------------------------------------------------------------

class NoisePredictor(nnx.Module):
    """Wraps a UNet so it predicts noise ``epsilon_theta(x_t, t, c)``."""

    def __init__(self, backbone: nnx.Module) -> None:
        self.backbone = backbone

    @typecheck
    def __call__(
        self,
        x_t: Float[Array, "B H W C"],
        t: Int[Array, "B"],
        cond: Optional[Float[Array, "..."]] = None,
    ) -> Float[Array, "B H W C"]:
        if cond is None:
            return self.backbone(x_t, t)
        return self.backbone(x_t, t, cond)


@typecheck
def classifier_free_guidance(
    eps_cond: Float[Array, "B H W C"],
    eps_uncond: Float[Array, "B H W C"],
    scale: float = 7.5,
) -> Float[Array, "B H W C"]:
    """``eps = eps_uncond + s * (eps_cond - eps_uncond)``."""
    return eps_uncond + scale * (eps_cond - eps_uncond)


# ---------------------------------------------------------------------------
# ControlNet
# ---------------------------------------------------------------------------

class ZeroConv(nnx.Conv):
    """1x1 conv initialized to zero - the connector ControlNet uses."""

    def __init__(self, in_ch: int, out_ch: int, *, rngs: nnx.Rngs) -> None:
        super().__init__(in_ch, out_ch, (1, 1),
                         kernel_init=nnx.initializers.zeros,
                         bias_init=nnx.initializers.zeros, rngs=rngs)


class ControlNetBlock(nnx.Module):
    """Toy ControlNet hint encoder - encodes a conditioning image and adds via ZeroConv."""

    def __init__(self, hint_ch: int, out_ch: int, hidden: int = 64,
                 *, rngs: nnx.Rngs) -> None:
        self.c1 = nnx.Conv(hint_ch, hidden, (3, 3), padding="SAME", rngs=rngs)
        self.c2 = nnx.Conv(hidden, hidden, (3, 3), strides=2,
                           padding="SAME", rngs=rngs)
        self.c3 = nnx.Conv(hidden, hidden * 2, (3, 3), padding="SAME", rngs=rngs)
        self.c4 = nnx.Conv(hidden * 2, hidden * 2, (3, 3), strides=2,
                           padding="SAME", rngs=rngs)
        self.zero = ZeroConv(hidden * 2, out_ch, rngs=rngs)

    @typecheck
    def __call__(
        self, hint: Float[Array, "B H W C_hint"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        h = nnx.silu(self.c1(hint))
        h = nnx.silu(self.c2(h))
        h = nnx.silu(self.c3(h))
        h = nnx.silu(self.c4(h))
        return self.zero(h)


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nnx.Module):
    """Low-rank adapter for an existing :class:`nnx.Linear`.

    The base layer is treated as frozen; only ``lora_A`` / ``lora_B`` are
    trained. Use ``nnx.split`` with a custom filter to keep only adapter
    params in the gradient computation.
    """

    def __init__(self, base: nnx.Linear, rank: int = 8, alpha: float = 16.0,
                 *, rngs: nnx.Rngs) -> None:
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        in_f = base.kernel.value.shape[0]
        out_f = base.kernel.value.shape[1]
        self.lora_A = nnx.Param(
            jax.random.normal(rngs.params(), (in_f, rank)) * (1.0 / rank ** 0.5))
        self.lora_B = nnx.Param(jnp.zeros((rank, out_f)))

    @typecheck
    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_out"]:
        return self.base(x) + (x @ self.lora_A.value) @ self.lora_B.value * self.scale


class LoRAConv2d(nnx.Module):
    """LoRA adapter for an :class:`nnx.Conv` (NHWC)."""

    def __init__(self, base: nnx.Conv, rank: int = 4, alpha: float = 8.0,
                 *, rngs: nnx.Rngs) -> None:
        self.base = base
        self.scale = alpha / rank
        in_ch = base.kernel.value.shape[-2]
        out_ch = base.kernel.value.shape[-1]
        ksize = base.kernel.value.shape[:-2]
        self.down = nnx.Conv(in_ch, rank, ksize, strides=base.strides,
                             padding=base.padding, use_bias=False, rngs=rngs)
        self.up = nnx.Conv(rank, out_ch, (1,) * len(ksize), use_bias=False,
                           kernel_init=nnx.initializers.zeros, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H_out W_out C_out"]:
        return self.base(x) + self.up(self.down(x)) * self.scale


# ---------------------------------------------------------------------------
# Hypernetwork
# ---------------------------------------------------------------------------

class HyperNetwork(nnx.Module):
    """Small network that generates ``in_dim -> out_dim`` weight matrices on-the-fly."""

    def __init__(self, code_dim: int, in_dim: int, out_dim: int,
                 hidden: int = 128, *, rngs: nnx.Rngs) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = nnx.Linear(code_dim, hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, in_dim * out_dim + out_dim, rngs=rngs)

    @typecheck
    def __call__(
        self,
        code: Float[Array, "*B D_code"],
        x: Float[Array, "*B D_in"],
    ) -> Float[Array, "*B D_out"]:
        params = self.fc2(nnx.gelu(self.fc1(code)))
        w = params[..., : self.in_dim * self.out_dim].reshape(
            *code.shape[:-1], self.out_dim, self.in_dim)
        b = params[..., self.in_dim * self.out_dim:]
        return jnp.einsum("...oi,...i->...o", w, x) + b


# ---------------------------------------------------------------------------
# IP-Adapter
# ---------------------------------------------------------------------------

class IPAdapterCrossAttention(nnx.Module):
    """Two-stream cross-attention: text + image-prompt features (Ye et al. 2023)."""

    def __init__(self, dim: int, text_dim: int, image_dim: int,
                 num_heads: int = 8, scale: float = 1.0,
                 *, rngs: nnx.Rngs) -> None:
        self.scale = scale
        self.text_attn = CrossAttention(dim, text_dim, num_heads, rngs=rngs)
        self.image_attn = CrossAttention(dim, image_dim, num_heads, rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B Tq D"],
        text: Float[Array, "B T_text D_text"],
        image: Float[Array, "B T_img D_img"],
    ) -> Float[Array, "B Tq D"]:
        return self.text_attn(x, text) + self.scale * self.image_attn(x, image)
