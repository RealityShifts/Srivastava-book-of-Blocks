"""Smoke-test the PyTorch tracer on the patterns that used to fabricate edges.

Run with ``python -m visualized._smoke_test`` from ``models/blocks``. Like the
``pytorch_blocks`` smoke test it is not a unit-test suite - one model, one real
forward pass, and a handful of asserts on the graph that comes out.

Every construct below once produced an ``inferred`` edge - one drawn from
execution order rather than tensor identity - in a *simple* model, each for a
different reason:

- a shared ``Linear`` called twice on strided slices ``x[..., 0::2, :]``: the
  slice was tagged for the op tracer only, never for the module hooks, so the
  ``Linear`` saw an input with no producer and the order-based fallback wired
  it to whatever leaf finished last (a different branch's ``Linear``), while
  the real input edge was missing entirely
- eval-mode ``Dropout`` returning its input object: correctly credited to the
  real producer, it had no consumer of its own, and the orphan repair invented
  ``Dropout -> <next module>`` around the residual add
- ``-x`` and ``.round().long()``: pointwise ops outside both op tables severed
  provenance, so the consumer after them was repaired past the ops between
- ``F.scaled_dot_product_attention``: untraced, its result was a value from
  nowhere and the ``transpose`` consuming it got a fabricated producer

A graph of this model must now contain **no** inferred edge at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .visualize_torch import trace_model
from .prune_torch import prune_passthrough


class PairPool(nn.Module):
    """Shared Linear on the two halves of a 2:1 interleaved sequence."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 4)

    def forward(self, a):
        h1, h2 = self.proj(a[..., 0::2, :]), self.proj(a[..., 1::2, :])
        if h2.shape[-2] < h1.shape[-2]:                 # odd length boundary
            h2 = torch.cat([h2, h1[..., -1:, :]], dim=-2)
        return (h1 + h2) / 2


class TinyAttention(nn.Module):
    """RoPE-style rotate (unary minus + cat) and a fused sdpa call."""

    def __init__(self, dim=4, heads=2):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.heads = heads

    def forward(self, x):
        b, t, d = x.shape
        split = lambda z: z.view(b, t, self.heads, d // self.heads).transpose(1, 2)
        q, k, v = split(self.q(x)), split(self.k(x)), split(self.v(x))
        q1, q2 = q.chunk(2, dim=-1)
        q = torch.cat([-q2, q1], dim=-1)                  # rotate_half
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(4)
        self.attn = TinyAttention()
        self.mlp = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4), nn.Dropout(0.1))

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x + self.mlp(x)                            # Dropout is identity in eval


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.motion_in = nn.Linear(3, 4)
        self.pair = PairPool()
        self.level = nn.Linear(1, 4)
        self.blocks = nn.ModuleList([Block(), Block()])
        self.out = nn.Linear(4, 3)

    def forward(self, x, a, t):
        t_idx = (t.clamp(0, 1) * 999).round().long()      # cast chain
        h = self.motion_in(x) + self.pair(a) + self.level(t_idx.float()[..., None])
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


def main() -> None:
    torch.manual_seed(0)
    model = Model().eval()
    x, a, t = torch.randn(2, 5, 3), torch.randn(2, 9, 8), torch.rand(2, 5)   # 9: odd audio length
    graph = trace_model(model, x, a, t)
    prune_passthrough(graph)
    assert graph.error is None, graph.error

    by_id = {n.id: n for n in graph.nodes}
    label = lambda i: by_id[i].path if i in by_id else f"<input {i}>"

    inferred = [(label(e.src), label(e.dst)) for e in graph.edges if e.inferred]
    assert not inferred, f"execution-order edges fabricated: {inferred}"

    live = {n.id for n in graph.nodes}
    dangling = [e for e in graph.edges
                if (e.src >= 0 and e.src not in live) or (e.dst >= 0 and e.dst not in live)]
    assert not dangling, f"unresolvable edge endpoints: {dangling}"

    proj_ids = [n.id for n in graph.nodes if n.path == "pair.proj"]
    assert len(proj_ids) == 2, "shared Linear should be recorded once per call"
    for pid in proj_ids:
        feeders = [e.src for e in graph.edges if e.dst == pid]
        assert feeders and all(s < 0 for s in feeders), (
            f"pair.proj should be fed by the model input only, got {[label(s) for s in feeders]}")
    motion_targets = [label(e.dst) for e in graph.edges if label(e.src) == "motion_in"]
    assert "pair.proj" not in motion_targets, "motion_in must not feed pair.proj"

    sdpa = [n for n in graph.nodes if n.op == "scaled_dot_product_attention"]
    assert len(sdpa) == 2, "one sdpa node per block expected"

    print(f"ok: {len(graph.nodes)} nodes, {len(graph.edges)} edges, 0 inferred, "
          f"{len(sdpa)} sdpa nodes, pair.proj fed by the input on both calls")


if __name__ == "__main__":
    main()
