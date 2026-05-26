"""Section 16 - Sparse / efficient ML blocks.

Symmetric INT8 / 4-bit quantized linear layers, magnitude-based weight
pruning, attention-score token pruning, low-rank factorization,
column-/row-parallel linear layers, and a pipeline-stage scaffold.

The parallel layers detect ``jax`` device meshes; in single-device mode
they degrade gracefully so the code is testable without a real cluster.
"""

from __future__ import annotations

from typing import Iterable

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float, Int8


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def quantize_int8(
    weight: Float[Array, "D_in D_out"]
) -> tuple[Int8[Array, "D_in D_out"], Float[Array, "D_out"]]:
    """Per-output-row symmetric INT8 quantization."""
    scale = jnp.maximum(jnp.max(jnp.abs(weight), axis=0, keepdims=True),
                        1e-8) / 127.0
    q = jnp.round(weight / scale).clip(-127, 127).astype(jnp.int8)
    return q, scale.squeeze(0)


def dequantize_int8(
    qweight: Int8[Array, "D_in D_out"],
    scale: Float[Array, "D_out"],
) -> Float[Array, "D_in D_out"]:
    return qweight.astype(scale.dtype) * scale[None, :]


class QuantizedLinearInt8(nnx.Module):
    """Simulated INT8 linear: stores INT8 weights, dequantizes for matmul."""

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.qweight = nnx.Variable(
            jnp.zeros((in_features, out_features), dtype=jnp.int8))
        self.scale = nnx.Variable(jnp.ones((out_features,)))
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None

    @classmethod
    def from_linear(cls, linear: nnx.Linear, *, rngs: nnx.Rngs
                    ) -> "QuantizedLinearInt8":
        in_f, out_f = linear.kernel.value.shape
        layer = cls(in_f, out_f, use_bias=linear.use_bias, rngs=rngs)
        q, s = quantize_int8(linear.kernel.value)
        layer.qweight.value = q
        layer.scale.value = s
        if linear.use_bias and layer.bias is not None:
            layer.bias.value = linear.bias.value
        return layer

    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_out"]:
        w = dequantize_int8(self.qweight.value, self.scale.value.astype(x.dtype))
        out = x @ w
        if self.bias is not None:
            out = out + self.bias.value
        return out


class QuantizedLinear4bit(nnx.Module):
    """Simulated 4-bit symmetric weight quantization, two values per byte."""

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        if in_features % 2:
            raise ValueError("in_features must be even for 4-bit packing")
        self.in_features = in_features
        self.out_features = out_features
        self.qweight = nnx.Variable(
            jnp.zeros((in_features // 2, out_features), dtype=jnp.uint8))
        self.scale = nnx.Variable(jnp.ones((out_features,)))
        self.bias = nnx.Param(jnp.zeros((out_features,))) if use_bias else None

    @classmethod
    def from_linear(cls, linear: nnx.Linear, *, rngs: nnx.Rngs
                    ) -> "QuantizedLinear4bit":
        in_f, out_f = linear.kernel.value.shape
        layer = cls(in_f, out_f, use_bias=linear.use_bias, rngs=rngs)
        w = linear.kernel.value
        scale = jnp.maximum(jnp.max(jnp.abs(w), axis=0, keepdims=True), 1e-8) / 7.0
        q = jnp.round(w / scale).clip(-8, 7).astype(jnp.int8) + 8
        packed = (q[0::2] | (q[1::2] << 4)).astype(jnp.uint8)
        layer.qweight.value = packed
        layer.scale.value = scale.squeeze(0)
        if linear.use_bias and layer.bias is not None:
            layer.bias.value = linear.bias.value
        return layer

    def _dequantize(
        self, dtype: jnp.dtype
    ) -> Float[Array, "D_in D_out"]:
        lo = (self.qweight.value & 0x0F).astype(jnp.int8) - 8
        hi = (self.qweight.value >> 4).astype(jnp.int8) - 8
        unpacked = jnp.empty((self.in_features, self.out_features), dtype=dtype)
        unpacked = unpacked.at[0::2].set(lo.astype(dtype))
        unpacked = unpacked.at[1::2].set(hi.astype(dtype))
        return unpacked * self.scale.value.astype(dtype)[None, :]

    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_out"]:
        out = x @ self._dequantize(x.dtype)
        if self.bias is not None:
            out = out + self.bias.value
        return out


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class MagnitudePruner:
    """Unstructured magnitude pruning.

    Calling :meth:`apply` zeroes out the smallest-|w| weights of every
    Linear / Conv kernel reachable from the supplied modules and stores
    a binary mask. :meth:`reapply` re-zeros pruned weights after a step.
    """

    def __init__(self, sparsity: float = 0.5) -> None:
        self.sparsity = sparsity
        self.masks: dict[int, Float[Array, "..."]] = {}
        self.params: dict[int, nnx.Param] = {}

    def apply(self, modules: Iterable[nnx.Module]) -> None:
        for m in modules:
            for _, value in nnx.iter_graph(m):
                if not isinstance(value, nnx.Module):
                    continue
                kernel = getattr(value, "kernel", None)
                if not isinstance(kernel, nnx.Param):
                    continue
                w = kernel.value
                k = int(self.sparsity * w.size)
                if k <= 0:
                    continue
                threshold = jnp.sort(jnp.abs(w).reshape(-1))[k - 1]
                mask = (jnp.abs(w) > threshold).astype(w.dtype)
                kernel.value = w * mask
                self.masks[id(kernel)] = mask
                self.params[id(kernel)] = kernel

    def reapply(self) -> None:
        for k, p in self.params.items():
            p.value = p.value * self.masks[k]


class TokenPruner(nnx.Module):
    """Drops the lowest-scoring tokens to a fixed keep ratio (DynamicViT-style)."""

    def __init__(self, keep_ratio: float = 0.7) -> None:
        if not 0 < keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        self.keep_ratio = keep_ratio

    def __call__(
        self,
        x: Float[Array, "B T D"],
        scores: Float[Array, "B T"],
    ) -> Float[Array, "B T_kept D"]:
        B, T, C = x.shape
        k = max(1, int(T * self.keep_ratio))
        idx = jnp.sort(jax.lax.top_k(scores, k)[1], axis=-1)
        return jnp.take_along_axis(x, idx[:, :, None], axis=1)


# ---------------------------------------------------------------------------
# Low-rank factorization
# ---------------------------------------------------------------------------

class LowRankLinear(nnx.Module):
    """Replaces a dense ``in -> out`` layer with two thin layers (rank ``r``)."""

    def __init__(self, in_features: int, out_features: int, rank: int,
                 use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        self.down = nnx.Linear(in_features, rank, use_bias=False, rngs=rngs)
        self.up = nnx.Linear(rank, out_features, use_bias=use_bias, rngs=rngs)

    @classmethod
    def from_linear(cls, linear: nnx.Linear, rank: int, *,
                    rngs: nnx.Rngs) -> "LowRankLinear":
        in_f, out_f = linear.kernel.value.shape
        layer = cls(in_f, out_f, rank, use_bias=linear.use_bias, rngs=rngs)
        u, s, vh = jnp.linalg.svd(linear.kernel.value, full_matrices=False)
        s_sqrt = jnp.sqrt(s[:rank])
        layer.down.kernel.value = u[:, :rank] * s_sqrt[None, :]   # (in, r)
        layer.up.kernel.value = vh[:rank] * s_sqrt[:, None]       # (r, out)
        if linear.use_bias and layer.up.use_bias:
            layer.up.bias.value = linear.bias.value
        return layer

    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_out"]:
        return self.up(self.down(x))


# ---------------------------------------------------------------------------
# Tensor / pipeline parallelism scaffolds
# ---------------------------------------------------------------------------

def _device_count() -> int:
    return jax.device_count()


class ColumnParallelLinear(nnx.Module):
    """Linear whose ``out_features`` are sharded across the tensor-parallel group.

    This implementation stores the *local* shard only; in a sharded run
    the next layer (typically :class:`RowParallelLinear`) finishes the
    matmul with an all-reduce.
    """

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        ws = _device_count()
        if out_features % ws:
            raise ValueError("out_features must be divisible by device count")
        self.local_out = out_features // ws
        self.fc = nnx.Linear(in_features, self.local_out, use_bias=use_bias,
                             rngs=rngs)

    def __call__(
        self, x: Float[Array, "*B D_in"]
    ) -> Float[Array, "*B D_local_out"]:
        return self.fc(x)


class RowParallelLinear(nnx.Module):
    """Linear whose ``in_features`` are sharded; performs all-reduce after matmul."""

    def __init__(self, in_features: int, out_features: int,
                 use_bias: bool = True, *, rngs: nnx.Rngs) -> None:
        ws = _device_count()
        if in_features % ws:
            raise ValueError("in_features must be divisible by device count")
        self.local_in = in_features // ws
        self.fc = nnx.Linear(self.local_in, out_features, use_bias=use_bias,
                             rngs=rngs)

    def __call__(
        self, x: Float[Array, "*B D_local_in"]
    ) -> Float[Array, "*B D_out"]:
        out = self.fc(x)
        if jax.device_count() > 1:
            out = jax.lax.psum(out, axis_name="tp")
        return out


class PipelineStage(nnx.Module):
    """A single stage of a pipeline: takes input from prev stage, sends to next."""

    def __init__(self, module: nnx.Module, stage_id: int,
                 num_stages: int) -> None:
        self.module = module
        self.stage_id = stage_id
        self.num_stages = num_stages

    def __call__(self, x: Float[Array, "..."]) -> Float[Array, "..."]:
        return self.module(x)
