"""Section 12 - Memory and retrieval blocks.

A differentiable external key/value memory, a retrieval-augmented
generation skeleton and a transformer KV-cache.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor


# ---------------------------------------------------------------------------
# Differentiable external memory
# ---------------------------------------------------------------------------

class ExternalMemory(nn.Module):
    """Soft addressable key/value memory (NTM/Memory-Networks flavour).

    Reads return a softmax-weighted sum of values, where the weights are
    cosine-similarity between query and keys.
    """

    def __init__(self, num_slots: int, key_dim: int, value_dim: int) -> None:
        super().__init__()
        self.keys = nn.Parameter(torch.randn(num_slots, key_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(num_slots, value_dim) * 0.02)

    def forward(
        self, query: Float[Tensor, "*B D_k"]
    ) -> Float[Tensor, "*B D_v"]:
        q = F.normalize(query, dim=-1)
        k = F.normalize(self.keys, dim=-1)
        weights = torch.softmax(q @ k.T, dim=-1)
        return weights @ self.values


# ---------------------------------------------------------------------------
# Retrieval-Augmented Generation
# ---------------------------------------------------------------------------

class VectorStore:
    """Tiny in-memory vector store with cosine top-k search."""

    def __init__(self) -> None:
        self.embeddings: Optional[Float[Tensor, "N D"]] = None
        self.documents: list = []

    def add(
        self, embeddings: Float[Tensor, "N D"], documents: list
    ) -> None:
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = torch.cat([self.embeddings, embeddings], dim=0)
        self.documents.extend(documents)

    def search(
        self, query: Float[Tensor, "Q D"], k: int = 4
    ) -> tuple[Float[Tensor, "Q K"], list]:
        assert self.embeddings is not None and len(self.documents) > 0
        q = F.normalize(query, dim=-1)
        e = F.normalize(self.embeddings, dim=-1)
        scores = q @ e.T
        topk = scores.topk(k, dim=-1)
        docs = [[self.documents[i] for i in row.tolist()] for row in topk.indices]
        return topk.values, docs


class RAGModule(nn.Module):
    """Retriever + generator container.

    The user supplies an *encoder* (text -> vector) and a *generator*
    (text -> text) callable; this class only orchestrates retrieval.
    """

    def __init__(self, encoder: Callable[[list[str]], Float[Tensor, "Q D"]],
                 generator: Callable[[str], str], top_k: int = 4) -> None:
        super().__init__()
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

    Stores keys/values of shape ``(B, H, T, D)``. Each call to :meth:`update`
    appends new tokens and returns the full cached tensors.
    """

    def __init__(self, num_layers: int) -> None:
        self.k: list[Optional[Float[Tensor, "B H T D"]]] = [None] * num_layers
        self.v: list[Optional[Float[Tensor, "B H T D"]]] = [None] * num_layers

    def update(
        self,
        layer: int,
        k: Float[Tensor, "B H T_new D"],
        v: Float[Tensor, "B H T_new D"],
    ) -> tuple[Float[Tensor, "B H T D"], Float[Tensor, "B H T D"]]:
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]                      # type: ignore[return-value]

    def length(self, layer: int = 0) -> int:
        return 0 if self.k[layer] is None else self.k[layer].shape[2]    # type: ignore[union-attr]

    def reset(self) -> None:
        for i in range(len(self.k)):
            self.k[i] = None
            self.v[i] = None
