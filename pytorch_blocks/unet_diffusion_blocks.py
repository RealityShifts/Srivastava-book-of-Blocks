"""Section 5 - UNet & diffusion blocks.

UNet (encoder-decoder + skips), down/up-sample blocks, sinusoidal time
embedding, classifier-free guidance helper, ControlNet-style conditioning,
LoRA, hypernetwork, IP-Adapter cross-attention.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from ._typecheck import typecheck
from torch import Tensor

from .attention_blocks import CrossAttention


def _gn_groups(channels: int, target: int = 32) -> int:
    """Pick a number of GroupNorm groups that evenly divides ``channels``."""
    for g in range(min(channels, target), 0, -1):
        if channels % g == 0:
            return g
    return 1


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Diffusion-style sinusoidal embedding of integer timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    @typecheck
    def forward(
        self, t: Int[Tensor, "B"]
    ) -> Float[Tensor, "B D"]:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepMLP(nn.Module):
    """Sinusoidal embedding followed by a 2-layer MLP - used by DDPM/SD."""

    def __init__(self, dim: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or 4 * dim
        self.embed = SinusoidalTimeEmbedding(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    @typecheck
    def forward(
        self, t: Int[Tensor, "B"]
    ) -> Float[Tensor, "B D_hidden"]:
        return self.net(self.embed(t))


# ---------------------------------------------------------------------------
# Down / up sample
# ---------------------------------------------------------------------------

class DownsampleBlock(nn.Module):
    """Strided 3x3 conv (or avg-pool) for halving spatial size."""

    def __init__(self, channels: int, mode: str = "conv") -> None:
        super().__init__()
        if mode == "conv":
            self.op: nn.Module = nn.Conv2d(channels, channels, 3, 2, 1)
        elif mode == "avg":
            self.op = nn.AvgPool2d(2)
        else:
            raise ValueError(mode)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H_out W_out"]:
        return self.op(x)


class UpsampleBlock(nn.Module):
    """Restores spatial resolution via interpolation, transposed conv, pixel shuffle, or blur.

    ``"blur"`` is the StyleGAN2 recipe: nearest-neighbour 2x, an anti-alias
    blur, then the conv. Worth preferring over ``"interp"`` in a generator - a
    bare nearest-neighbour upsample injects high-frequency stair-stepping that
    the following conv then bakes in as texture, and the blur is what removes
    it before that happens.
    """

    def __init__(self, channels: int, mode: str = "interp") -> None:
        super().__init__()
        if mode == "interp":
            self.op: nn.Module = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(channels, channels, 3, 1, 1),
            )
        elif mode == "transpose":
            self.op = nn.ConvTranspose2d(channels, channels, 4, 2, 1)
        elif mode == "shuffle":
            self.op = nn.Sequential(
                nn.Conv2d(channels, channels * 4, 3, 1, 1),
                nn.PixelShuffle(2),
            )
        elif mode == "blur":
            from .filters import Blur2d       # local: keeps filters optional
            self.op = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                Blur2d(channels),
                nn.Conv2d(channels, channels, 3, 1, 1),
            )
        else:
            raise ValueError(mode)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H_out W_out"]:
        return self.op(x)


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNetResBlock(nn.Module):
    """Diffusion-style residual block conditioned on a time embedding."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.t_proj = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B C_in H W"],
        t_emb: Float[Tensor, "B D_t"],
    ) -> Float[Tensor, "B C_out H W"]:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class UNet(nn.Module):
    """A compact diffusion-style UNet.

    Parameters
    ----------
    in_ch, out_ch:  input/output channel counts (e.g. 4 for SD latents)
    base:           channel multiplier for the stem
    ch_mults:       relative channel multipliers per resolution
    t_dim:          time embedding dimension
    """

    def __init__(self, in_ch: int = 4, out_ch: int = 4, base: int = 64,
                 ch_mults: Sequence[int] = (1, 2, 4, 4), t_dim: int = 256) -> None:
        super().__init__()
        self.t_embed = TimestepMLP(t_dim)

        self.stem = nn.Conv2d(in_ch, base, 3, 1, 1)

        self.downs = nn.ModuleList()
        chans = [base]
        c = base
        for i, m in enumerate(ch_mults):
            o = base * m
            self.downs.append(nn.ModuleList([
                UNetResBlock(c, o, 4 * t_dim),
                UNetResBlock(o, o, 4 * t_dim),
                DownsampleBlock(o) if i < len(ch_mults) - 1 else nn.Identity(),
            ]))
            chans.extend([o, o])
            c = o

        self.mid1 = UNetResBlock(c, c, 4 * t_dim)
        self.mid2 = UNetResBlock(c, c, 4 * t_dim)

        self.ups = nn.ModuleList()
        for i, m in enumerate(reversed(ch_mults)):
            o = base * m
            skip_c = chans.pop()
            skip_c2 = chans.pop()
            self.ups.append(nn.ModuleList([
                UNetResBlock(c + skip_c, o, 4 * t_dim),
                UNetResBlock(o + skip_c2, o, 4 * t_dim),
                UpsampleBlock(o) if i < len(ch_mults) - 1 else nn.Identity(),
            ]))
            c = o

        self.out_norm = nn.GroupNorm(_gn_groups(c), c)
        self.out_conv = nn.Conv2d(c, out_ch, 3, 1, 1)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B C_in H W"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B C_out H W"]:
        t_emb = self.t_embed(t)
        h = self.stem(x)
        skips = [h]
        for r1, r2, down in self.downs:
            h = r1(h, t_emb); skips.append(h)
            h = r2(h, t_emb); skips.append(h)
            h = down(h)
        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)
        for r1, r2, up in self.ups:
            h = r1(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = r2(torch.cat([h, skips.pop()], dim=1), t_emb)
            h = up(h)
        return self.out_conv(F.silu(self.out_norm(h)))


# ---------------------------------------------------------------------------
# Noise prediction wrapper + classifier-free guidance
# ---------------------------------------------------------------------------

class NoisePredictor(nn.Module):
    """Wraps a UNet so it predicts noise ``epsilon_theta(x_t, t, c)``."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    @typecheck
    def forward(
        self,
        x_t: Float[Tensor, "B C H W"],
        t: Int[Tensor, "B"],
        cond: Optional[Float[Tensor, "..."]] = None,
    ) -> Float[Tensor, "B C H W"]:
        if cond is None:
            return self.backbone(x_t, t)
        return self.backbone(x_t, t, cond)


@typecheck
def classifier_free_guidance(
    eps_cond: Float[Tensor, "B C H W"],
    eps_uncond: Float[Tensor, "B C H W"],
    scale: float = 7.5,
) -> Float[Tensor, "B C H W"]:
    """``eps = eps_uncond + s * (eps_cond - eps_uncond)``."""
    return eps_uncond + scale * (eps_cond - eps_uncond)


# ---------------------------------------------------------------------------
# ControlNet
# ---------------------------------------------------------------------------

class ZeroConv2d(nn.Conv2d):
    """1x1 conv initialized to zero - the connector ControlNet uses."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(in_ch, out_ch, 1)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class ControlNetBlock(nn.Module):
    """Toy ControlNet hint encoder - encodes control image and adds via ZeroConv."""

    def __init__(self, hint_ch: int, out_ch: int, hidden: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(hint_ch, hidden, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, stride=2), nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, 3, padding=1), nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, stride=2), nn.SiLU(),
        )
        self.zero = ZeroConv2d(hidden * 2, out_ch)

    @typecheck
    def forward(
        self, hint: Float[Tensor, "B C_hint H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return self.zero(self.encoder(hint))


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter for a frozen ``nn.Linear``: ``W' = W + (B A) * alpha/r``."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        self.drop = nn.Dropout(dropout)

    @typecheck
    def forward(
        self, x: Float[Tensor, "*B D_in"]
    ) -> Float[Tensor, "*B D_out"]:
        return self.base(x) + self.drop(x) @ self.lora_A.T @ self.lora_B.T * self.scale


class LoRAConv2d(nn.Module):
    """LoRA wrapper for a frozen :class:`nn.Conv2d`."""

    def __init__(self, base: nn.Conv2d, rank: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        for p in base.parameters():
            p.requires_grad_(False)
        self.base = base
        self.scale = alpha / rank
        self.down = nn.Conv2d(base.in_channels, rank, base.kernel_size,
                              stride=base.stride, padding=base.padding, bias=False)
        self.up = nn.Conv2d(rank, base.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=5 ** 0.5)
        nn.init.zeros_(self.up.weight)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        return self.base(x) + self.up(self.down(x)) * self.scale


# ---------------------------------------------------------------------------
# Hypernetwork
# ---------------------------------------------------------------------------

class HyperNetwork(nn.Module):
    """Small network that generates ``in_dim -> out_dim`` weight matrices on-the-fly."""

    def __init__(self, code_dim: int, in_dim: int, out_dim: int,
                 hidden: int = 128) -> None:
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.net = nn.Sequential(
            nn.Linear(code_dim, hidden), nn.GELU(),
            nn.Linear(hidden, in_dim * out_dim + out_dim),
        )

    @typecheck
    def forward(
        self,
        code: Float[Tensor, "*B D_code"],
        x: Float[Tensor, "*B D_in"],
    ) -> Float[Tensor, "*B D_out"]:
        params = self.net(code)
        w = params[..., : self.in_dim * self.out_dim].view(
            *code.shape[:-1], self.out_dim, self.in_dim)
        b = params[..., self.in_dim * self.out_dim:]
        return torch.einsum("...oi,...i->...o", w, x) + b


# ---------------------------------------------------------------------------
# IP-Adapter
# ---------------------------------------------------------------------------

class IPAdapterCrossAttention(nn.Module):
    """Two-stream cross-attention: text + image-prompt features (Ye et al. 2023)."""

    def __init__(self, dim: int, text_dim: int, image_dim: int,
                 num_heads: int = 8, scale: float = 1.0) -> None:
        super().__init__()
        self.scale = scale
        self.text_attn = CrossAttention(dim, text_dim, num_heads)
        self.image_attn = CrossAttention(dim, image_dim, num_heads)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B T D"],
        text: Float[Tensor, "B T_txt D_txt"],
        image: Float[Tensor, "B T_img D_img"],
    ) -> Float[Tensor, "B T D"]:
        return self.text_attn(x, text) + self.scale * self.image_attn(x, image)
