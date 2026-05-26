"""Section 15 - Multimodal & agentic blocks.

CLIP-style dual encoder, the Perceiver Resampler used by Flamingo,
the Q-Former from BLIP-2, a tool-use dispatcher, memory-attention.
"""

from __future__ import annotations

import re
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from ._typecheck import typecheck

from .attention_blocks import CrossAttention, MultiHeadAttention
from .transformer_blocks import FeedForward


# ---------------------------------------------------------------------------
# CLIP-style dual encoder
# ---------------------------------------------------------------------------

class CLIPEncoder(nnx.Module):
    """Maps a vision and a text tower into a joint L2-normalized space."""

    def __init__(self, vision_encoder: nnx.Module, text_encoder: nnx.Module,
                 vision_dim: int, text_dim: int, embed_dim: int = 512,
                 *, rngs: nnx.Rngs) -> None:
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_proj = nnx.Linear(vision_dim, embed_dim,
                                      use_bias=False, rngs=rngs)
        self.text_proj = nnx.Linear(text_dim, embed_dim,
                                    use_bias=False, rngs=rngs)

    def encode_image(
        self, image: Float[Array, "B H W C"]
    ) -> Float[Array, "B D_embed"]:
        z = self.vision_proj(self.vision_encoder(image))
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)

    def encode_text(
        self, text: Float[Array, "B T D_text"]
    ) -> Float[Array, "B D_embed"]:
        z = self.text_proj(self.text_encoder(text))
        return z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + 1e-8)


# ---------------------------------------------------------------------------
# Perceiver Resampler (Flamingo)
# ---------------------------------------------------------------------------

class PerceiverResampler(nnx.Module):
    """Compresses a long sequence of input tokens into ``num_latents`` queries."""

    def __init__(self, dim: int, num_latents: int = 64, num_heads: int = 8,
                 depth: int = 6, mlp_ratio: float = 4.0,
                 *, rngs: nnx.Rngs) -> None:
        self.latents = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_latents, dim)) * 0.02)
        layers = []
        for _ in range(depth):
            layers.append((
                nnx.LayerNorm(dim, rngs=rngs),
                CrossAttention(dim, dim, num_heads, rngs=rngs),
                nnx.LayerNorm(dim, rngs=rngs),
                FeedForward(dim, int(dim * mlp_ratio), rngs=rngs),
            ))
        self.layers = layers
        self.norm = nnx.LayerNorm(dim, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B K D"]:
        B = x.shape[0]
        latents = jnp.broadcast_to(self.latents.value,
                                   (B, *self.latents.value.shape[1:]))
        for n1, attn, n2, ffn in self.layers:
            kv = jnp.concatenate([n1(x), n1(latents)], axis=1)
            latents = latents + attn(n1(latents), kv)
            latents = latents + ffn(n2(latents))
        return self.norm(latents)


# ---------------------------------------------------------------------------
# Q-Former (BLIP-2)
# ---------------------------------------------------------------------------

class QFormer(nnx.Module):
    """Bridges a frozen image encoder and a frozen LLM via learnable queries."""

    def __init__(self, dim: int = 768, num_queries: int = 32, num_heads: int = 12,
                 depth: int = 6, image_dim: Optional[int] = None,
                 llm_dim: int = 4096, *, rngs: nnx.Rngs) -> None:
        image_dim = image_dim or dim
        self.queries = nnx.Param(
            jax.random.normal(rngs.params(), (1, num_queries, dim)) * 0.02)
        layers = []
        for _ in range(depth):
            layers.append((
                nnx.LayerNorm(dim, rngs=rngs),
                MultiHeadAttention(dim, num_heads, rngs=rngs),
                nnx.LayerNorm(dim, rngs=rngs),
                CrossAttention(dim, image_dim, num_heads, rngs=rngs),
                nnx.LayerNorm(dim, rngs=rngs),
                FeedForward(dim, 4 * dim, rngs=rngs),
            ))
        self.layers = layers
        self.norm = nnx.LayerNorm(dim, rngs=rngs)
        self.proj = nnx.Linear(dim, llm_dim, rngs=rngs)

    @typecheck
    def __call__(
        self, image_feats: Float[Array, "B T D_img"]
    ) -> Float[Array, "B K D_llm"]:
        B = image_feats.shape[0]
        q = jnp.broadcast_to(self.queries.value,
                             (B, *self.queries.value.shape[1:]))
        for n1, sa, n2, ca, n3, ffn in self.layers:
            q = q + sa(n1(q))
            q = q + ca(n2(q), image_feats)
            q = q + ffn(n3(q))
        return self.proj(self.norm(q))


# ---------------------------------------------------------------------------
# Tool use
# ---------------------------------------------------------------------------

class ToolUseBlock:
    """Lightweight tool dispatcher.

    Parses ``<tool name="..">arg</tool>`` markers from a model's text output
    and routes them to a registered Python callable.
    """

    def __init__(self) -> None:
        self.tools: dict[str, Callable[[str], str]] = {}

    def register(self, name: str, fn: Callable[[str], str]) -> None:
        self.tools[name] = fn

    @typecheck
    def __call__(self, text: str) -> str:
        out, pos = [], 0
        for m in re.finditer(r'<tool name="([^"]+)">([^<]*)</tool>', text):
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

class MemoryAttention(nnx.Module):
    """Cross-attention over an external memory bank (e.g. retrieved facts)."""

    def __init__(self, dim: int, mem_dim: Optional[int] = None,
                 num_heads: int = 8, *, rngs: nnx.Rngs) -> None:
        self.attn = CrossAttention(dim, mem_dim, num_heads, rngs=rngs)
        self.norm_q = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_kv = nnx.LayerNorm(mem_dim or dim, rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "B Tq D"],
        memory: Float[Array, "B Tm D_mem"],
    ) -> Float[Array, "B Tq D"]:
        return x + self.attn(self.norm_q(x), self.norm_kv(memory))
