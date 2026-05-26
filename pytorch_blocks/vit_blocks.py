"""Section 7 - Vision Transformer blocks.

Patch embedding, CLS token, Swin window / shifted-window attention,
masked-image-modeling head (MAE/BEiT).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor


# ---------------------------------------------------------------------------
# Patch + CLS
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Image -> sequence of patch tokens via a strided conv."""

    def __init__(self, in_ch: int = 3, patch_size: int = 16, dim: int = 768) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, patch_size, patch_size)

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B N D"]:
        x = self.proj(x)                                       # (B, D, H/p, W/p)
        return x.flatten(2).transpose(1, 2)                    # (B, N, D)


class CLSToken(nn.Module):
    """Learnable [CLS] token prepended to each sequence."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(
        self, x: Float[Tensor, "B N D"]
    ) -> Float[Tensor, "B N1 D"]:
        cls = self.token.expand(x.shape[0], -1, -1)
        return torch.cat([cls, x], dim=1)


# ---------------------------------------------------------------------------
# Swin window attention
# ---------------------------------------------------------------------------

def _window_partition(
    x: Float[Tensor, "B H W C"], w: int
) -> Float[Tensor, "BnW WW C"]:
    """``(B, H, W, C) -> (B*nW, w*w, C)``."""
    B, H, W, C = x.shape
    x = x.view(B, H // w, w, W // w, w, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, w * w, C)


def _window_reverse(
    windows: Float[Tensor, "BnW WW C"], w: int, H: int, W: int
) -> Float[Tensor, "B H W C"]:
    B = windows.shape[0] // (H * W // (w * w))
    x = windows.view(B, H // w, W // w, w, w, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)


class SwinWindowAttention(nn.Module):
    """Standard non-shifted window attention used inside Swin blocks."""

    def __init__(self, dim: int, num_heads: int, window: int = 7,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.window = window
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

        self.rel_bias = nn.Parameter(torch.zeros((2 * window - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window), torch.arange(window), indexing="ij"
        )).flatten(1)                                          # (2, w*w)
        rel = coords[:, :, None] - coords[:, None, :]          # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()
        rel[..., 0] += window - 1
        rel[..., 1] += window - 1
        rel[..., 0] *= 2 * window - 1
        self.register_buffer("rel_index", rel.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

    def forward(
        self,
        x: Float[Tensor, "BnW N D"],
        mask: Optional[Float[Tensor, "nW N N"]] = None,
    ) -> Float[Tensor, "BnW N D"]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q * self.head_dim ** -0.5) @ k.transpose(-2, -1)

        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1).permute(2, 0, 1)
        attn = attn + bias[None]

        if mask is not None:                                    # (nW, N, N)
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N)
            attn = attn + mask[None, :, None]
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.drop(self.proj(out))


class ShiftedWindowAttention(nn.Module):
    """Cyclic-shift variant enabling cross-window communication (Swin)."""

    def __init__(self, dim: int, input_resolution: tuple[int, int],
                 num_heads: int, window: int = 7, shift: Optional[int] = None,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.window = window
        self.shift = shift if shift is not None else window // 2
        self.attn = SwinWindowAttention(dim, num_heads, window, dropout)

        if self.shift > 0:
            H, W = input_resolution
            img_mask = torch.zeros(1, H, W, 1)
            cnt = 0
            for h_slice in (slice(0, -window), slice(-window, -self.shift),
                            slice(-self.shift, None)):
                for w_slice in (slice(0, -window), slice(-window, -self.shift),
                                slice(-self.shift, None)):
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, window).squeeze(-1)
            attn_mask = mask_windows[:, None] - mask_windows[:, :, None]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
            self.register_buffer("attn_mask", attn_mask, persistent=False)
        else:
            self.attn_mask = None

    def forward(
        self, x: Float[Tensor, "B N D"]
    ) -> Float[Tensor, "B N D"]:
        B, N, C = x.shape
        H, W = self.input_resolution
        assert N == H * W
        x = x.view(B, H, W, C)
        if self.shift > 0:
            x = torch.roll(x, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = _window_partition(x, self.window)
        attn = self.attn(windows, mask=self.attn_mask)
        x = _window_reverse(attn, self.window, H, W)
        if self.shift > 0:
            x = torch.roll(x, shifts=(self.shift, self.shift), dims=(1, 2))
        return x.view(B, N, C)


# ---------------------------------------------------------------------------
# Masked image modeling
# ---------------------------------------------------------------------------

class MaskedImageModeling(nn.Module):
    """MAE / BEiT-style head: mask a fraction of tokens, reconstruct them."""

    def __init__(self, dim: int, patch_size: int = 16, in_ch: int = 3,
                 mask_ratio: float = 0.75) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.decoder = nn.Linear(dim, patch_size * patch_size * in_ch)

    def random_masking(
        self, x: Float[Tensor, "B N D"]
    ) -> tuple[
        Float[Tensor, "B N_keep D"],
        Int[Tensor, "B N"],
        Float[Tensor, "B N"],
    ]:
        """Returns ``(visible_tokens, restore_indices, mask)``."""
        B, N, _ = x.shape
        keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        x_visible = torch.gather(x, 1, ids_keep[:, :, None].expand(-1, -1, x.shape[-1]))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_visible, ids_restore, mask

    def reconstruct(
        self,
        encoded: Float[Tensor, "B N_keep D"],
        ids_restore: Int[Tensor, "B N"],
    ) -> Float[Tensor, "B N P"]:
        B, N = ids_restore.shape
        n_mask = N - encoded.shape[1]
        tokens = torch.cat([encoded, self.mask_token.expand(B, n_mask, -1)], dim=1)
        tokens = torch.gather(tokens, 1,
                              ids_restore[:, :, None].expand(-1, -1, tokens.shape[-1]))
        return self.decoder(tokens)
