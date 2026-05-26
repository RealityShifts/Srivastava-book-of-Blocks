"""Section 9 - Graph Neural Network blocks.

Generic message-passing scaffold, the GCN graph-convolution layer (Kipf
& Welling 2016) and the Graph Attention Network layer (Velickovic 2018).

Graphs are represented in *edge-list* form: ``edge_index`` is a
``LongTensor`` of shape ``(2, E)`` with rows ``[src, dst]``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# Message passing scaffold
# ---------------------------------------------------------------------------

class MessagePassing(nn.Module):
    """Generic message passing: ``out = aggregate(message(x_src, x_dst, e))``.

    Subclasses override :meth:`message` (and optionally :meth:`update`).
    """

    aggregator: str = "sum"

    def message(
        self,
        x_src: Float[Tensor, "E D"],
        x_dst: Float[Tensor, "E D"],
        edge_attr: Optional[Float[Tensor, "E D_e"]] = None,
    ) -> Float[Tensor, "E D"]:
        return x_src

    def update(
        self,
        aggr: Float[Tensor, "N D"],
        x: Float[Tensor, "N D_in"],
    ) -> Float[Tensor, "N D"]:
        return aggr

    def aggregate(
        self,
        msg: Float[Tensor, "E D"],
        dst: Int[Tensor, "E"],
        num_nodes: int,
    ) -> Float[Tensor, "N D"]:
        out = torch.zeros(num_nodes, msg.shape[-1],
                          device=msg.device, dtype=msg.dtype)
        if self.aggregator == "sum":
            return out.index_add_(0, dst, msg)
        if self.aggregator == "mean":
            count = torch.zeros(num_nodes, 1, device=msg.device, dtype=msg.dtype)
            count.index_add_(0, dst, torch.ones_like(msg[:, :1]))
            out.index_add_(0, dst, msg)
            return out / count.clamp(min=1)
        if self.aggregator == "max":
            out.fill_(float("-inf"))
            out.scatter_reduce_(0, dst[:, None].expand_as(msg), msg,
                                reduce="amax", include_self=True)
            return out.masked_fill(out == float("-inf"), 0.0)
        raise ValueError(self.aggregator)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "N D_in"],
        edge_index: Int[Tensor, "2 E"],
        edge_attr: Optional[Float[Tensor, "E D_e"]] = None,
    ) -> Float[Tensor, "N D"]:
        src, dst = edge_index
        msg = self.message(x[src], x[dst], edge_attr)
        aggr = self.aggregate(msg, dst, x.shape[0])
        return self.update(aggr, x)


# ---------------------------------------------------------------------------
# Graph Convolution (GCN)
# ---------------------------------------------------------------------------

class GraphConv(MessagePassing):
    """Symmetric-normalized GCN layer: ``X' = D^{-1/2} A D^{-1/2} X W``."""

    aggregator = "sum"

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=True)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "N D_in"],
        edge_index: Int[Tensor, "2 E"],
        edge_attr: Optional[Float[Tensor, "E D_e"]] = None,
    ) -> Float[Tensor, "N D_out"]:
        x = self.lin(x)
        src, dst = edge_index
        deg = torch.zeros(x.shape[0], device=x.device).index_add_(
            0, dst, torch.ones_like(src, dtype=x.dtype))
        norm = deg.clamp(min=1).pow(-0.5)
        weight = norm[src] * norm[dst]
        msg = x[src] * weight[:, None]
        out = torch.zeros_like(x).index_add_(0, dst, msg)
        return out


# ---------------------------------------------------------------------------
# GAT
# ---------------------------------------------------------------------------

class GraphAttention(nn.Module):
    """Multi-head Graph Attention layer (Velickovic 2018)."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4,
                 dropout: float = 0.0, negative_slope: float = 0.2) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self.slope = negative_slope
        self.drop = nn.Dropout(dropout)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "N D_in"],
        edge_index: Int[Tensor, "2 E"],
    ) -> Float[Tensor, "N D_out"]:
        N = x.shape[0]
        h = self.lin(x).view(N, self.heads, self.out_dim)
        src, dst = edge_index
        alpha_src = (h * self.att_src).sum(-1)                 # (N, H)
        alpha_dst = (h * self.att_dst).sum(-1)
        e = F.leaky_relu(alpha_src[src] + alpha_dst[dst], self.slope)

        e = e - torch.full((N, self.heads), -1e30,
                           device=e.device).scatter_reduce_(
            0, dst[:, None].expand_as(e), e, reduce="amax", include_self=True)[dst]
        weights = e.exp()
        denom = torch.zeros(N, self.heads, device=x.device).index_add_(0, dst, weights)
        weights = weights / denom[dst].clamp(min=1e-16)
        weights = self.drop(weights)

        out = torch.zeros(N, self.heads, self.out_dim, device=x.device)
        out.index_add_(0, dst, h[src] * weights[:, :, None])
        return out.reshape(N, self.heads * self.out_dim)
