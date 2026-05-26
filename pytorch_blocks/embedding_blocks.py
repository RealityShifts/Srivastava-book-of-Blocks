"""Section 13 - Embedding & representation blocks.

Token embedding, learned and sinusoidal positional embeddings,
contrastive (InfoNCE) loss, and the projection head used by SimCLR/CLIP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# Token embedding
# ---------------------------------------------------------------------------

class TokenEmbedding(nn.Embedding):
    """Standard learned word embedding with truncated-normal init."""

    def __init__(self, vocab_size: int, dim: int, padding_idx: int | None = None):
        super().__init__(vocab_size, dim, padding_idx=padding_idx)
        nn.init.trunc_normal_(self.weight, std=0.02)
        if padding_idx is not None:
            with torch.no_grad():
                self.weight[padding_idx].zero_()


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class LearnedPositionalEmbedding(nn.Module):
    """Standard nn.Embedding-based positional encoding (BERT-style)."""

    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(max_len, dim)
        nn.init.trunc_normal_(self.embed.weight, std=0.02)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        T = x.shape[1]
        pos = torch.arange(T, device=x.device)
        return x + self.embed(pos)


class SinusoidalPositionalEmbedding(nn.Module):
    """Vaswani et al. 2017 - non-learned sinusoidal positions."""

    def __init__(self, max_len: int, dim: int) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10_000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        return x + self.pe[:, : x.shape[1]]


# ---------------------------------------------------------------------------
# Contrastive learning
# ---------------------------------------------------------------------------

class ProjectionHead(nn.Module):
    """SimCLR-style 2-layer MLP projection head."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    @typecheck
    def forward(
        self, x: Float[Tensor, "B D_in"]
    ) -> Float[Tensor, "B D_out"]:
        return self.net(x)


@typecheck
def info_nce(
    z1: Float[Tensor, "B D"],
    z2: Float[Tensor, "B D"],
    temperature: float = 0.07,
) -> Float[Tensor, ""]:
    """Symmetric InfoNCE / NT-Xent loss used by SimCLR & CLIP."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.shape[0], device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


class CLIPLoss(nn.Module):
    """CLIP loss with a learnable temperature ``logit_scale``."""

    def __init__(self, init_temperature: float = 0.07) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    @typecheck
    def forward(
        self,
        image_emb: Float[Tensor, "B D"],
        text_emb: Float[Tensor, "B D"],
    ) -> Float[Tensor, ""]:
        image_emb = F.normalize(image_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        logits = scale * image_emb @ text_emb.T
        labels = torch.arange(logits.shape[0], device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
