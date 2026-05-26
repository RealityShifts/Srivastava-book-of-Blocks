"""Section 13 - Embedding and representation blocks.

Token embedding, learned and sinusoidal positional embeddings,
contrastive InfoNCE loss, and the SimCLR/CLIP projection head.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int
from ._typecheck import typecheck


# ---------------------------------------------------------------------------
# Token embedding
# ---------------------------------------------------------------------------

TokenEmbedding = nnx.Embed  # ``num_embeddings -> features`` lookup


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class LearnedPositionalEmbedding(nnx.Module):
    """Standard learned positional encoding (BERT-style)."""

    def __init__(self, max_len: int, dim: int, *, rngs: nnx.Rngs) -> None:
        self.embed = nnx.Embed(max_len, dim, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        T = x.shape[1]
        pos = jnp.arange(T)
        return x + self.embed(pos)


class SinusoidalPositionalEmbedding(nnx.Module):
    """Vaswani et al. 2017 - non-learned sinusoidal positions."""

    def __init__(self, max_len: int, dim: int) -> None:
        pe = jnp.zeros((max_len, dim))
        pos = jnp.arange(max_len)[:, None]
        div = jnp.exp(jnp.arange(0, dim, 2) * (-math.log(10_000.0) / dim))
        pe = pe.at[:, 0::2].set(jnp.sin(pos * div))
        pe = pe.at[:, 1::2].set(jnp.cos(pos * div))
        self.pe = pe[None]

    @typecheck
    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        return x + self.pe[:, : x.shape[1]]


# ---------------------------------------------------------------------------
# Contrastive learning
# ---------------------------------------------------------------------------

class ProjectionHead(nnx.Module):
    """SimCLR-style 2-layer MLP projection head."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int,
                 *, rngs: nnx.Rngs) -> None:
        self.fc1 = nnx.Linear(in_dim, hidden, use_bias=False, rngs=rngs)
        self.bn = nnx.BatchNorm(hidden, rngs=rngs)
        self.fc2 = nnx.Linear(hidden, out_dim, use_bias=False, rngs=rngs)

    @typecheck
    def __call__(
        self, x: Float[Array, "B D_in"]
    ) -> Float[Array, "B D_out"]:
        return self.fc2(nnx.relu(self.bn(self.fc1(x))))


@typecheck
def info_nce(
    z1: Float[Array, "B D"],
    z2: Float[Array, "B D"],
    temperature: float = 0.07,
) -> Float[Array, ""]:
    """Symmetric InfoNCE / NT-Xent loss used by SimCLR & CLIP."""
    z1 = z1 / (jnp.linalg.norm(z1, axis=-1, keepdims=True) + 1e-8)
    z2 = z2 / (jnp.linalg.norm(z2, axis=-1, keepdims=True) + 1e-8)
    logits = z1 @ z2.T / temperature
    labels = jnp.arange(z1.shape[0])
    return 0.5 * (
        _cross_entropy(logits, labels) + _cross_entropy(logits.T, labels))


def _cross_entropy(
    logits: Float[Array, "B C"],
    labels: Int[Array, "B"],
) -> Float[Array, ""]:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.mean(jnp.take_along_axis(log_probs, labels[:, None], axis=-1))


class CLIPLoss(nnx.Module):
    """CLIP loss with a learnable temperature ``logit_scale``."""

    def __init__(self, init_temperature: float = 0.07) -> None:
        self.logit_scale = nnx.Param(
            jnp.log(jnp.array(1.0 / init_temperature)))

    @typecheck
    def __call__(
        self,
        image_emb: Float[Array, "B D"],
        text_emb: Float[Array, "B D"],
    ) -> Float[Array, ""]:
        image_emb = image_emb / (jnp.linalg.norm(image_emb, axis=-1,
                                                 keepdims=True) + 1e-8)
        text_emb = text_emb / (jnp.linalg.norm(text_emb, axis=-1,
                                               keepdims=True) + 1e-8)
        scale = jnp.minimum(jnp.exp(self.logit_scale.value), 100.0)
        logits = scale * image_emb @ text_emb.T
        labels = jnp.arange(logits.shape[0])
        return 0.5 * (
            _cross_entropy(logits, labels) + _cross_entropy(logits.T, labels))
