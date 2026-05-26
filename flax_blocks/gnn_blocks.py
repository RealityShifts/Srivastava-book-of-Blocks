"""Section 9 - Graph Neural Network blocks.

Generic message passing, GCN graph-convolution (Kipf & Welling 2016),
multi-head Graph Attention layer (Velickovic 2018).

Graphs are represented in *edge-list* form: ``edge_index`` is an
integer ``jax.Array`` of shape ``(2, E)`` with rows ``[src, dst]``.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int
from ._typecheck import typecheck


# ---------------------------------------------------------------------------
# Message passing
# ---------------------------------------------------------------------------

@typecheck
def scatter_sum(
    msg: Float[Array, "E D"],
    dst: Int[Array, "E"],
    num_nodes: int,
) -> Float[Array, "N D"]:
    out = jnp.zeros((num_nodes, msg.shape[-1]), dtype=msg.dtype)
    return out.at[dst].add(msg)


@typecheck
def scatter_mean(
    msg: Float[Array, "E D"],
    dst: Int[Array, "E"],
    num_nodes: int,
) -> Float[Array, "N D"]:
    summed = scatter_sum(msg, dst, num_nodes)
    counts = jnp.zeros((num_nodes,), dtype=msg.dtype).at[dst].add(1.0)
    return summed / jnp.maximum(counts, 1)[:, None]


@typecheck
def scatter_max(
    msg: Float[Array, "E D"],
    dst: Int[Array, "E"],
    num_nodes: int,
) -> Float[Array, "N D"]:
    init = jnp.full((num_nodes, msg.shape[-1]), -jnp.inf, dtype=msg.dtype)
    out = init.at[dst].max(msg)
    return jnp.where(jnp.isinf(out), 0.0, out)


class MessagePassing(nnx.Module):
    """Generic message passing scaffold; subclass and override ``message``."""

    aggregator: str = "sum"

    def message(
        self,
        x_src: Float[Array, "E D"],
        x_dst: Float[Array, "E D"],
        edge_attr: Optional[Float[Array, "E D_edge"]] = None,
    ) -> Float[Array, "E D_msg"]:
        return x_src

    def update(
        self,
        aggr: Float[Array, "N D_msg"],
        x: Float[Array, "N D"],
    ) -> Float[Array, "N D_out"]:
        return aggr

    def aggregate(
        self,
        msg: Float[Array, "E D"],
        dst: Int[Array, "E"],
        num_nodes: int,
    ) -> Float[Array, "N D"]:
        if self.aggregator == "sum":
            return scatter_sum(msg, dst, num_nodes)
        if self.aggregator == "mean":
            return scatter_mean(msg, dst, num_nodes)
        if self.aggregator == "max":
            return scatter_max(msg, dst, num_nodes)
        raise ValueError(self.aggregator)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "N D"],
        edge_index: Int[Array, "2 E"],
        edge_attr: Optional[Float[Array, "E D_edge"]] = None,
    ) -> Float[Array, "N D_out"]:
        src, dst = edge_index
        msg = self.message(x[src], x[dst], edge_attr)
        aggr = self.aggregate(msg, dst, x.shape[0])
        return self.update(aggr, x)


# ---------------------------------------------------------------------------
# Graph Convolution
# ---------------------------------------------------------------------------

class GraphConv(nnx.Module):
    """Symmetric-normalized GCN: ``X' = D^{-1/2} A D^{-1/2} X W``."""

    def __init__(self, in_dim: int, out_dim: int, *, rngs: nnx.Rngs) -> None:
        self.lin = nnx.Linear(in_dim, out_dim, rngs=rngs)

    @typecheck
    def __call__(
        self,
        x: Float[Array, "N D_in"],
        edge_index: Int[Array, "2 E"],
    ) -> Float[Array, "N D_out"]:
        x = self.lin(x)
        src, dst = edge_index
        deg = jnp.zeros(x.shape[0]).at[dst].add(1.0)
        norm = jnp.power(jnp.maximum(deg, 1), -0.5)
        weight = norm[src] * norm[dst]
        msg = x[src] * weight[:, None]
        return scatter_sum(msg, dst, x.shape[0])


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------

class GraphAttention(nnx.Module):
    """Multi-head Graph Attention layer (Velickovic 2018)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 negative_slope: float = 0.2, *, rngs: nnx.Rngs) -> None:
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nnx.Linear(in_dim, heads * out_dim, use_bias=False, rngs=rngs)
        self.att_src = nnx.Param(
            jax.random.normal(rngs.params(), (1, heads, out_dim)) * 0.1)
        self.att_dst = nnx.Param(
            jax.random.normal(rngs.params(), (1, heads, out_dim)) * 0.1)
        self.slope = negative_slope

    @typecheck
    def __call__(
        self,
        x: Float[Array, "N D_in"],
        edge_index: Int[Array, "2 E"],
    ) -> Float[Array, "N D_out_times_H"]:
        N = x.shape[0]
        h = self.lin(x).reshape(N, self.heads, self.out_dim)
        src, dst = edge_index
        alpha_src = jnp.sum(h * self.att_src.value, axis=-1)             # (N, H)
        alpha_dst = jnp.sum(h * self.att_dst.value, axis=-1)
        e = nnx.leaky_relu(alpha_src[src] + alpha_dst[dst], self.slope)

        max_per_node = jnp.full((N, self.heads), -jnp.inf).at[dst].max(e)
        e = e - max_per_node[dst]
        weights = jnp.exp(e)
        denom = jnp.zeros((N, self.heads)).at[dst].add(weights)
        weights = weights / jnp.maximum(denom[dst], 1e-16)

        out = jnp.zeros((N, self.heads, self.out_dim)).at[dst].add(
            h[src] * weights[:, :, None])
        return out.reshape(N, self.heads * self.out_dim)
