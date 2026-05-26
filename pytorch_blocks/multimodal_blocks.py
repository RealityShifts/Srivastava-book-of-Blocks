"""Section 15 - Multimodal & agentic blocks.

CLIP-style dual encoder, the Perceiver Resampler used by Flamingo,
the Q-Former from BLIP-2, a tool-use dispatcher, and memory-attention.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from jaxtyping import Float
from ._typecheck import typecheck
from torch import Tensor

from .attention_blocks import CrossAttention, MultiHeadAttention
from .transformer_blocks import TransformerEncoderBlock, FeedForward


# ---------------------------------------------------------------------------
# CLIP-style dual encoder
# ---------------------------------------------------------------------------

class CLIPEncoder(nn.Module):
    """A dual encoder that maps a vision and a text tower into a joint space.

    The actual vision/text encoders are user-supplied; the head is a linear
    projection per modality plus L2 normalization of the output features.
    """

    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module,
                 vision_dim: int, text_dim: int, embed_dim: int = 512) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_proj = nn.Linear(vision_dim, embed_dim, bias=False)
        self.text_proj = nn.Linear(text_dim, embed_dim, bias=False)

    def encode_image(
        self, image: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B D"]:
        return torch.nn.functional.normalize(
            self.vision_proj(self.vision_encoder(image)), dim=-1)

    def encode_text(
        self, text: Tensor
    ) -> Float[Tensor, "B D"]:
        return torch.nn.functional.normalize(
            self.text_proj(self.text_encoder(text)), dim=-1)


# ---------------------------------------------------------------------------
# Perceiver Resampler (Flamingo)
# ---------------------------------------------------------------------------

class PerceiverResampler(nn.Module):
    """Compresses a long sequence of input tokens into ``num_latents`` queries."""

    def __init__(self, dim: int, num_latents: int = 64, num_heads: int = 8,
                 depth: int = 6, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                CrossAttention(dim, dim, num_heads),
                nn.LayerNorm(dim),
                FeedForward(dim, int(dim * mlp_ratio)),
            ]))
        self.norm = nn.LayerNorm(dim)

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B K D"]:
        B = x.shape[0]
        latents = self.latents.expand(B, -1, -1)
        for n1, attn, n2, ffn in self.layers:
            kv = torch.cat([n1(x), n1(latents)], dim=1)
            latents = latents + attn(n1(latents), kv)
            latents = latents + ffn(n2(latents))
        return self.norm(latents)


# ---------------------------------------------------------------------------
# Q-Former (BLIP-2)
# ---------------------------------------------------------------------------

class QFormer(nn.Module):
    """Bridges a frozen image encoder and a frozen LLM via learnable query tokens.

    Each layer alternates self-attention over the queries with cross-attention
    into the image features. The output queries are projected to the LLM's
    embedding dimension.
    """

    def __init__(self, dim: int = 768, num_queries: int = 32,
                 num_heads: int = 12, depth: int = 6,
                 image_dim: Optional[int] = None,
                 llm_dim: int = 4096) -> None:
        super().__init__()
        image_dim = image_dim or dim
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                MultiHeadAttention(dim, num_heads),
                nn.LayerNorm(dim),
                CrossAttention(dim, image_dim, num_heads),
                nn.LayerNorm(dim),
                FeedForward(dim, 4 * dim),
            ]))
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, llm_dim)

    @typecheck
    def forward(
        self, image_feats: Float[Tensor, "B T D_img"]
    ) -> Float[Tensor, "B Q D_llm"]:
        B = image_feats.shape[0]
        q = self.queries.expand(B, -1, -1)
        for n1, sa, n2, ca, n3, ffn in self.layers:
            q = q + sa(n1(q))
            q = q + ca(n2(q), image_feats)
            q = q + ffn(n3(q))
        return self.proj(self.norm(q))


# ---------------------------------------------------------------------------
# Tool use
# ---------------------------------------------------------------------------

class ToolUseBlock(nn.Module):
    """Lightweight tool dispatcher.

    Parses a sentinel token from the model's text stream of the form
    ``<tool name="..">arg</tool>`` and routes it to a registered Python callable.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tools: dict[str, Callable[[str], str]] = {}

    def register(self, name: str, fn: Callable[[str], str]) -> None:
        self.tools[name] = fn

    @typecheck
    def __call__(self, text: str) -> str:
        import re
        out, pos = [], 0
        for m in re.finditer(r"<tool name=\"([^\"]+)\">([^<]*)</tool>", text):
            out.append(text[pos: m.start()])
            tool, arg = m.group(1), m.group(2)
            if tool in self.tools:
                out.append(self.tools[tool](arg))
            pos = m.end()
        out.append(text[pos:])
        return "".join(out)


# ---------------------------------------------------------------------------
# Memory attention
# ---------------------------------------------------------------------------

class MemoryAttention(nn.Module):
    """Cross-attention over an external memory bank (e.g. retrieved facts)."""

    def __init__(self, dim: int, mem_dim: Optional[int] = None,
                 num_heads: int = 8) -> None:
        super().__init__()
        self.attn = CrossAttention(dim, mem_dim, num_heads)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(mem_dim or dim)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B T D"],
        memory: Float[Tensor, "B M D_mem"],
    ) -> Float[Tensor, "B T D"]:
        return x + self.attn(self.norm_q(x), self.norm_kv(memory))
