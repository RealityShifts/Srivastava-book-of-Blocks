"""Section 16 - Sparse / efficient ML blocks.

Symmetric INT8 and 4-bit quantized linear layers, magnitude-based
weight pruning, attention-score token pruning, low-rank factorization,
column- and row-parallel linear layers and a pipeline-stage scaffold.

The parallel layers fall back to single-process execution when
``torch.distributed`` is not initialized, so they're usable for
exposition and unit tests without a real cluster.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from jaxtyping import Float, Int8
from torch import Tensor


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_int8(
    weight: Float[Tensor, "O I"],
) -> tuple[Int8[Tensor, "O I"], Float[Tensor, "O"]]:
    """Per-output-row symmetric INT8 quantization, returning ``(qweight, scale)``."""
    scale = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(weight / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(-1)


def dequantize_int8(
    qweight: Int8[Tensor, "O I"], scale: Float[Tensor, "O"]
) -> Float[Tensor, "O I"]:
    return qweight.to(scale.dtype) * scale[:, None]


class QuantizedLinearInt8(nn.Module):
    """Simulated INT8 linear: stores INT8 weights, dequantizes for matmul."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight",
                             torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer("scale", torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinearInt8":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None)
        q, s = quantize_int8(linear.weight.data)
        layer.qweight.copy_(q)
        layer.scale.copy_(s)
        if linear.bias is not None and layer.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def forward(
        self, x: Float[Tensor, "*B I"]
    ) -> Float[Tensor, "*B O"]:
        w = dequantize_int8(self.qweight, self.scale.to(x.dtype))
        return F.linear(x, w, self.bias)


class QuantizedLinear4bit(nn.Module):
    """Simulated 4-bit symmetric weight quantization stored two-per-byte."""

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        if in_features % 2:
            raise ValueError("in_features must be even for 4-bit packing")
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight",
                             torch.zeros(out_features, in_features // 2,
                                         dtype=torch.uint8))
        self.register_buffer("scale", torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "QuantizedLinear4bit":
        layer = cls(linear.in_features, linear.out_features, linear.bias is not None)
        w = linear.weight.data
        scale = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 7.0
        q = torch.round(w / scale).clamp(-8, 7).to(torch.int8) + 8       # range [0, 15]
        packed = (q[:, 0::2] | (q[:, 1::2] << 4)).to(torch.uint8)
        layer.qweight.copy_(packed)
        layer.scale.copy_(scale.squeeze(-1))
        if linear.bias is not None and layer.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def _dequantize(
        self, dtype: torch.dtype
    ) -> Float[Tensor, "O I"]:
        lo = (self.qweight & 0x0F).to(torch.int8) - 8
        hi = (self.qweight >> 4).to(torch.int8) - 8
        unpacked = torch.empty(self.out_features, self.in_features,
                               device=self.qweight.device, dtype=dtype)
        unpacked[:, 0::2] = lo.to(dtype)
        unpacked[:, 1::2] = hi.to(dtype)
        return unpacked * self.scale.to(dtype)[:, None]

    def forward(
        self, x: Float[Tensor, "*B I"]
    ) -> Float[Tensor, "*B O"]:
        return F.linear(x, self._dequantize(x.dtype), self.bias)


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class MagnitudePruner:
    """Unstructured magnitude pruning by replacing the smallest-|w| weights with 0."""

    def __init__(self, sparsity: float = 0.5) -> None:
        self.sparsity = sparsity
        self.masks: dict[nn.Parameter, Float[Tensor, "..."]] = {}

    def apply(self, modules: Iterable[nn.Module]) -> None:
        for m in modules:
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                w = m.weight.data
                k = int(self.sparsity * w.numel())
                if k <= 0:
                    continue
                threshold = w.abs().flatten().kthvalue(k).values
                mask = (w.abs() > threshold).to(w.dtype)
                w.mul_(mask)
                self.masks[m.weight] = mask

    def reapply(self) -> None:
        """Re-zero masked weights after a gradient step."""
        for p, mask in self.masks.items():
            p.data.mul_(mask)


class TokenPruner(nn.Module):
    """Drops the lowest-scoring tokens to a fixed keep ratio (DynamicViT-style)."""

    def __init__(self, keep_ratio: float = 0.7) -> None:
        super().__init__()
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio

    def forward(
        self,
        x: Float[Tensor, "B T D"],
        scores: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B T_kept D"]:
        B, T, C = x.shape
        k = max(1, int(T * self.keep_ratio))
        idx = scores.topk(k, dim=-1).indices.sort(-1).values
        idx = idx[:, :, None].expand(-1, -1, C)
        return torch.gather(x, 1, idx)


# ---------------------------------------------------------------------------
# Low-Rank factorization
# ---------------------------------------------------------------------------

class LowRankLinear(nn.Module):
    """Replaces a dense ``in_features x out_features`` layer with two thin layers."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> "LowRankLinear":
        layer = cls(linear.in_features, linear.out_features, rank,
                    linear.bias is not None)
        u, s, vh = torch.linalg.svd(linear.weight.data, full_matrices=False)
        s_sqrt = s[:rank].sqrt()
        layer.down.weight.data.copy_(vh[:rank] * s_sqrt[:, None])
        layer.up.weight.data.copy_(u[:, :rank] * s_sqrt[None, :])
        if linear.bias is not None and layer.up.bias is not None:
            layer.up.bias.data.copy_(linear.bias.data)
        return layer

    def forward(
        self, x: Float[Tensor, "*B I"]
    ) -> Float[Tensor, "*B O"]:
        return self.up(self.down(x))


# ---------------------------------------------------------------------------
# Tensor / Pipeline parallelism scaffolds
# ---------------------------------------------------------------------------

def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


class ColumnParallelLinear(nn.Module):
    """Linear whose ``out_features`` are sharded across the tensor-parallel group.

    Output is *not* gathered; the next layer should be a :class:`RowParallelLinear`
    that performs an all-reduce.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        ws = _world_size()
        if out_features % ws:
            raise ValueError("out_features must be divisible by world_size")
        self.local_out = out_features // ws
        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        self.bias = nn.Parameter(torch.zeros(self.local_out)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(
        self, x: Float[Tensor, "*B I"]
    ) -> Float[Tensor, "*B O_local"]:
        return F.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """Linear whose ``in_features`` are sharded; performs all-reduce after matmul."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        ws = _world_size()
        if in_features % ws:
            raise ValueError("in_features must be divisible by world_size")
        self.local_in = in_features // ws
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

    def forward(
        self, x: Float[Tensor, "*B I_local"]
    ) -> Float[Tensor, "*B O"]:
        out = F.linear(x, self.weight)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out


class PipelineStage(nn.Module):
    """A single stage of a pipeline: takes input from prev stage, sends to next."""

    def __init__(self, module: nn.Module, stage_id: int, num_stages: int) -> None:
        super().__init__()
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages

    def forward(
        self, x: Float[Tensor, "..."]
    ) -> Float[Tensor, "..."]:
        return self.module(x)
