"""Section 12 - Memory and retrieval blocks.

A differentiable external key/value memory, a tiny vector store + RAG
skeleton, and a transformer KV cache for fast LLM decoding.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# Differentiable external memory
# ---------------------------------------------------------------------------

class ExternalMemory(nnx.Module):
    """Soft addressable key/value memory (NTM/Memory-Networks flavour)."""

    def __init__(self, num_slots: int, key_dim: int, value_dim: int,
                 *, rngs: nnx.Rngs) -> None:
        self.keys = nnx.Param(
            jax.random.normal(rngs.params(), (num_slots, key_dim)) * 0.02)
        self.values = nnx.Param(
            jax.random.normal(rngs.params(), (num_slots, value_dim)) * 0.02)

    def __call__(
        self, query: Float[Array, "*B D_key"]
    ) -> Float[Array, "*B D_value"]:
        q = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
        k = self.keys.value / (jnp.linalg.norm(self.keys.value, axis=-1,
                                               keepdims=True) + 1e-8)
        weights = jax.nn.softmax(q @ k.T, axis=-1)
        return weights @ self.values.value


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

class VectorStore:
    """Tiny in-memory vector store with cosine top-k search."""

    def __init__(self) -> None:
        self.embeddings: Optional[Float[Array, "N D"]] = None
        self.documents: list = []

    def add(
        self,
        embeddings: Float[Array, "N D"],
        documents: list,
    ) -> None:
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = jnp.concatenate([self.embeddings, embeddings], axis=0)
        self.documents.extend(documents)

    def search(
        self, query: Float[Array, "Q D"], k: int = 4
    ) -> tuple[Float[Array, "Q K"], list]:
        assert self.embeddings is not None and self.documents
        q = query / (jnp.linalg.norm(query, axis=-1, keepdims=True) + 1e-8)
        e = self.embeddings / (jnp.linalg.norm(self.embeddings, axis=-1,
                                               keepdims=True) + 1e-8)
        scores = q @ e.T
        topk_v, topk_i = jax.lax.top_k(scores, k)
        docs = [[self.documents[int(i)] for i in row] for row in topk_i]
        return topk_v, docs


class RAGModule:
    """Retriever + generator container.

    The user supplies an *encoder* (text -> vector) and a *generator*
    (text -> text); this class only orchestrates retrieval.
    """

    def __init__(
        self,
        encoder: Callable[[list[str]], Float[Array, "N D"]],
        generator: Callable[[str], str],
        top_k: int = 4,
    ) -> None:
        self.encoder = encoder
        self.generator = generator
        self.store = VectorStore()
        self.top_k = top_k

    def index(self, docs: list[str]) -> None:
        self.store.add(self.encoder(docs), docs)

    def __call__(self, queries: list[str]) -> list[str]:
        emb = self.encoder(queries)
        _, retrieved = self.store.search(emb, k=self.top_k)
        out = []
        for q, ctx in zip(queries, retrieved):
            prompt = "Context:\n" + "\n".join(ctx) + f"\n\nQuestion: {q}\nAnswer:"
            out.append(self.generator(prompt))
        return out


# ---------------------------------------------------------------------------
# KV cache
# ---------------------------------------------------------------------------

class KVCache:
    """Per-layer growable cache for transformer decoding.

    Stores keys/values of shape ``(B, T, H, D)`` (Flax convention). Each
    call to :meth:`update` appends new tokens and returns the full cache.
    """

    def __init__(self, num_layers: int) -> None:
        self.k: list[Optional[Float[Array, "B T H D"]]] = [None] * num_layers
        self.v: list[Optional[Float[Array, "B T H D"]]] = [None] * num_layers

    def update(
        self,
        layer: int,
        k: Float[Array, "B T H D"],
        v: Float[Array, "B T H D"],
    ) -> tuple[Float[Array, "B T_total H D"], Float[Array, "B T_total H D"]]:
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = jnp.concatenate([self.k[layer], k], axis=1)
            self.v[layer] = jnp.concatenate([self.v[layer], v], axis=1)
        return self.k[layer], self.v[layer]                                  # type: ignore[return-value]

    def length(self, layer: int = 0) -> int:
        return 0 if self.k[layer] is None else self.k[layer].shape[1]        # type: ignore[union-attr]

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None
