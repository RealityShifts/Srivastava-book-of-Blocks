"""Section 4 - CNN and vision blocks.

Inception, DenseNet, Squeeze-and-Excitation, CBAM, Spatial Pyramid
Pooling, Feature Pyramid Network, ASPP, PixelShuffle, deformable
conv / deformable attention.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

from .core_blocks import ConvBlock


# ---------------------------------------------------------------------------
# Inception
# ---------------------------------------------------------------------------

class InceptionBlock(nn.Module):
    """Naive Inception-v1 module: 1x1, 3x3, 5x5 and pool branches."""

    def __init__(self, in_ch: int, c1: int = 64, c3: int = 96,
                 c5: int = 16, pool: int = 32) -> None:
        super().__init__()
        self.b1 = ConvBlock(in_ch, c1, 1, activation="relu")
        self.b3 = nn.Sequential(
            ConvBlock(in_ch, c3, 1, activation="relu"),
            ConvBlock(c3, c3, 3, activation="relu"),
        )
        self.b5 = nn.Sequential(
            ConvBlock(in_ch, c5, 1, activation="relu"),
            ConvBlock(c5, c5, 5, activation="relu"),
        )
        self.bp = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBlock(in_ch, pool, 1, activation="relu"),
        )

    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


# ---------------------------------------------------------------------------
# DenseNet
# ---------------------------------------------------------------------------

class _DenseLayer(nn.Sequential):
    def __init__(self, in_ch: int, growth: int) -> None:
        super().__init__(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 4 * growth, 1, bias=False),
            nn.BatchNorm2d(4 * growth), nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth, growth, 3, padding=1, bias=False),
        )

    def forward(                                                       # type: ignore[override]
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        return torch.cat([x, super().forward(x)], dim=1)


class DenseBlock(nn.Sequential):
    """A dense block with ``num_layers`` densely-connected conv units."""

    def __init__(self, in_ch: int, num_layers: int = 4, growth: int = 32) -> None:
        layers = []
        c = in_ch
        for _ in range(num_layers):
            layers.append(_DenseLayer(c, growth))
            c += growth
        super().__init__(*layers)
        self.out_channels = c


# ---------------------------------------------------------------------------
# Channel / spatial attention
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Hu et al. 2017 - SE channel attention."""

    def __init__(self, channels: int, ratio: int = 16) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1), nn.Sigmoid(),
        )

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        return x * self.fc(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module - channel + spatial attention."""

    def __init__(self, channels: int, ratio: int = 16, kernel: int = 7) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, channels))
        self.spatial = nn.Conv2d(2, 1, kernel, padding=kernel // 2)

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        ch = torch.sigmoid(self.mlp(avg) + self.mlp(mx))[:, :, None, None]
        x = x * ch
        sp = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial(sp))


# ---------------------------------------------------------------------------
# Multi-scale pooling / fusion
# ---------------------------------------------------------------------------

class SpatialPyramidPooling(nn.Module):
    """Multi-scale pyramid average pooling, output is flattened."""

    def __init__(self, output_sizes: Sequence[int] = (1, 2, 4)) -> None:
        super().__init__()
        self.pools = nn.ModuleList(nn.AdaptiveAvgPool2d(s) for s in output_sizes)

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B D_out"]:
        return torch.cat([p(x).flatten(1) for p in self.pools], dim=1)


class FeaturePyramidNetwork(nn.Module):
    """Top-down FPN with lateral 1x1 connections (Lin et al. 2017)."""

    def __init__(self, in_channels: Sequence[int], out_ch: int = 256) -> None:
        super().__init__()
        self.lat = nn.ModuleList(nn.Conv2d(c, out_ch, 1) for c in in_channels)
        self.smooth = nn.ModuleList(nn.Conv2d(out_ch, out_ch, 3, padding=1)
                                    for _ in in_channels)

    def forward(
        self, feats: Sequence[Float[Tensor, "B C H W"]]
    ) -> list[Float[Tensor, "B C_out H_l W_l"]]:
        lats = [lat(f) for lat, f in zip(self.lat, feats)]
        outs = [lats[-1]]
        for i in range(len(lats) - 2, -1, -1):
            up = F.interpolate(outs[-1], size=lats[i].shape[-2:], mode="nearest")
            outs.append(lats[i] + up)
        outs = outs[::-1]
        return [s(o) for s, o in zip(self.smooth, outs)]


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling - DeepLab v3."""

    def __init__(self, in_ch: int, out_ch: int = 256,
                 dilations: Sequence[int] = (1, 6, 12, 18)) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            k = 1 if d == 1 else 3
            self.branches.append(ConvBlock(in_ch, out_ch, k, dilation=d))
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(in_ch, out_ch, 1),
        )
        self.project = ConvBlock(out_ch * (len(dilations) + 1), out_ch, 1)

    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        size = x.shape[-2:]
        feats = [b(x) for b in self.branches]
        ip = F.interpolate(self.image_pool(x), size=size, mode="bilinear",
                           align_corners=False)
        return self.project(torch.cat(feats + [ip], dim=1))


# ---------------------------------------------------------------------------
# Up-/down-sampling
# ---------------------------------------------------------------------------

class PixelShuffleUpsample(nn.Module):
    """Sub-pixel convolution: ESPCN / SRResNet upsampler."""

    def __init__(self, channels: int, scale: int = 2) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, padding=1)
        self.shuffle = nn.PixelShuffle(scale)

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H_out W_out"]:
        return self.shuffle(self.conv(x))


# ---------------------------------------------------------------------------
# Deformable conv & deformable attention (simplified)
# ---------------------------------------------------------------------------

class DeformableConv2d(nn.Module):
    """Deformable convolution v1 (Dai et al. 2017) implemented with grid_sample.

    Slower than the CUDA op shipped in torchvision but pure-PyTorch.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = padding
        self.offset = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size,
                                kernel_size, stride, padding)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size,
                              stride=kernel_size, padding=0)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H_out W_out"]:
        B, C, H, W = x.shape
        k = self.k
        offset = self.offset(x)                                  # (B, 2k^2, Hout, Wout)
        Hout, Wout = offset.shape[-2:]

        ys, xs = torch.meshgrid(
            torch.arange(Hout, device=x.device, dtype=x.dtype),
            torch.arange(Wout, device=x.device, dtype=x.dtype),
            indexing="ij",
        )
        ky, kx = torch.meshgrid(
            torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2,
            torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2,
            indexing="ij",
        )
        base_y = ys[:, :, None, None] * self.stride + ky[None, None] - self.padding
        base_x = xs[:, :, None, None] * self.stride + kx[None, None] - self.padding
        dy, dx = offset.view(B, 2, k * k, Hout, Wout).chunk(2, dim=1)
        dy = dy.squeeze(1).permute(0, 2, 3, 1).reshape(B, Hout, Wout, k, k)
        dx = dx.squeeze(1).permute(0, 2, 3, 1).reshape(B, Hout, Wout, k, k)
        sample_y = base_y + dy
        sample_x = base_x + dx
        norm_y = 2.0 * sample_y / max(H - 1, 1) - 1.0
        norm_x = 2.0 * sample_x / max(W - 1, 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1).reshape(B, Hout * k, Wout * k, 2)
        sampled = F.grid_sample(x, grid, mode="bilinear",
                                padding_mode="zeros", align_corners=True)
        return self.conv(sampled)


class DeformableAttention(nn.Module):
    """Multi-head deformable attention (Zhu et al. - Deformable DETR).

    Each query produces ``num_points`` sampling offsets per head and pools
    values from those locations. Operates on a flat token grid ``(B, T=H*W, C)``.
    """

    def __init__(self, dim: int, num_heads: int = 8, num_points: int = 4) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.heads = num_heads
        self.points = num_points
        self.head_dim = dim // num_heads
        self.value = nn.Linear(dim, dim)
        self.offset = nn.Linear(dim, num_heads * num_points * 2)
        self.weight = nn.Linear(dim, num_heads * num_points)
        self.out = nn.Linear(dim, dim)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(
        self,
        x: Float[Tensor, "B T D"],
        hw: tuple[int, int],
    ) -> Float[Tensor, "B T D"]:
        B, T, C = x.shape
        H, W = hw
        v = self.value(x).transpose(1, 2).view(B, C, H, W)        # value map
        off = self.offset(x).view(B, T, self.heads, self.points, 2)
        w = self.weight(x).view(B, T, self.heads, self.points).softmax(-1)

        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device),
            torch.linspace(-1, 1, W, device=x.device),
            indexing="ij",
        )
        ref = torch.stack([xs, ys], dim=-1).reshape(T, 2)
        sample = ref[None, :, None, None, :] + off                # (B,T,h,p,2)

        v_h = v.view(B, self.heads, self.head_dim, H, W)
        outs = []
        for h in range(self.heads):
            grid = sample[:, :, h].reshape(B, T * self.points, 1, 2)
            sampled = F.grid_sample(v_h[:, h], grid,
                                    mode="bilinear", padding_mode="zeros",
                                    align_corners=True)             # (B,D,T*P,1)
            sampled = sampled.squeeze(-1).view(B, self.head_dim, T, self.points)
            agg = (sampled * w[:, :, h, :].unsqueeze(1)).sum(-1)     # (B,D,T)
            outs.append(agg.transpose(1, 2))                         # (B,T,D)
        return self.out(torch.cat(outs, dim=-1))                     # (B,T,C)
