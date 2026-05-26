"""Section 17 - Specialized blocks.

Neural ODE (explicit Euler), 2-D Fourier Neural Operator (FNO),
Kolmogorov-Arnold Network layer (RBF-based for differentiability),
Capsule layer + dynamic routing, Slot Attention.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Complex, Float, PRNGKeyArray


# ---------------------------------------------------------------------------
# Neural ODE
# ---------------------------------------------------------------------------

class NeuralODE(nnx.Module):
    """Continuous-depth network: integrate ``dh/dt = f_theta(h, t)`` with explicit Euler."""

    def __init__(self, dynamics: nnx.Module, num_steps: int = 16,
                 t0: float = 0.0, t1: float = 1.0) -> None:
        self.dynamics = dynamics
        self.num_steps = num_steps
        self.t0 = t0
        self.t1 = t1

    def __call__(
        self, x: Float[Array, "B *features"]
    ) -> Float[Array, "B *features"]:
        dt = (self.t1 - self.t0) / self.num_steps
        t = jnp.full((x.shape[0],), self.t0, dtype=x.dtype)
        h = x
        for _ in range(self.num_steps):
            h = h + dt * self.dynamics(h, t)
            t = t + dt
        return h


# ---------------------------------------------------------------------------
# Fourier Neural Operator
# ---------------------------------------------------------------------------

class SpectralConv2d(nnx.Module):
    """The core spectral conv used inside FNO (NHWC layout).

    Multiplies the lowest-frequency Fourier modes by a learned complex
    weight tensor and zero-keeps the higher modes.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_h: int, modes_w: int,
                 *, rngs: nnx.Rngs) -> None:
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mh = modes_h
        self.mw = modes_w
        scale = 1.0 / (in_ch * out_ch)
        shape = (modes_h, modes_w, in_ch, out_ch)
        self.w1_re = nnx.Param(jax.random.normal(rngs.params(), shape) * scale)
        self.w1_im = nnx.Param(jax.random.normal(rngs.params(), shape) * scale)
        self.w2_re = nnx.Param(jax.random.normal(rngs.params(), shape) * scale)
        self.w2_im = nnx.Param(jax.random.normal(rngs.params(), shape) * scale)

    def _w(
        self, re: nnx.Param, im: nnx.Param,
    ) -> Complex[Array, "mh mw C_in C_out"]:
        return re.value + 1j * im.value

    @staticmethod
    def _compl_mul(
        a: Complex[Array, "B mh mw C_in"],
        b: Complex[Array, "mh mw C_in C_out"],
    ) -> Complex[Array, "B mh mw C_out"]:
        return jnp.einsum("bxyi,xyio->bxyo", a, b)

    def __call__(
        self, x: Float[Array, "B H W C_in"]
    ) -> Float[Array, "B H W C_out"]:
        B, H, W, _ = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(1, 2), norm="ortho")
        out_ft = jnp.zeros((B, H, W // 2 + 1, self.out_ch), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :self.mh, :self.mw].set(
            self._compl_mul(x_ft[:, :self.mh, :self.mw],
                            self._w(self.w1_re, self.w1_im)))
        out_ft = out_ft.at[:, -self.mh:, :self.mw].set(
            self._compl_mul(x_ft[:, -self.mh:, :self.mw],
                            self._w(self.w2_re, self.w2_im)))
        return jnp.fft.irfft2(out_ft, s=(H, W), axes=(1, 2), norm="ortho")


class FNOBlock(nnx.Module):
    """Spectral conv + 1x1 residual + GELU (Li et al. 2020)."""

    def __init__(self, channels: int, modes_h: int = 12, modes_w: int = 12,
                 *, rngs: nnx.Rngs) -> None:
        self.spectral = SpectralConv2d(channels, channels, modes_h, modes_w,
                                       rngs=rngs)
        self.skip = nnx.Conv(channels, channels, (1, 1), rngs=rngs)

    def __call__(
        self, x: Float[Array, "B H W C"]
    ) -> Float[Array, "B H W C"]:
        return nnx.gelu(self.spectral(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Kolmogorov-Arnold Network layer
# ---------------------------------------------------------------------------

class KANLayer(nnx.Module):
    """A single layer of a Kolmogorov-Arnold Network (Liu et al. 2024).

    Each edge ``(i, j)`` carries a learnable function approximated as
    ``b * silu(x) + sum_k coeff_k * RBF_k(x)`` over fixed grid points.
    The RBF basis is a differentiable substitute for B-splines.
    """

    def __init__(self, in_dim: int, out_dim: int, num_grid: int = 8,
                 grid_range: tuple[float, float] = (-1.0, 1.0),
                 *, rngs: nnx.Rngs) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grid = num_grid
        self.grid = jnp.linspace(grid_range[0], grid_range[1], num_grid)
        self.coeff = nnx.Param(
            jax.random.normal(rngs.params(), (out_dim, in_dim, num_grid)) * 0.1)
        self.base_w = nnx.Param(
            jax.random.normal(rngs.params(), (in_dim, out_dim)) * 0.1)
        self.sigma = (grid_range[1] - grid_range[0]) / (num_grid - 1)

    def _basis(
        self, x: Float[Array, "B D_in"]
    ) -> Float[Array, "B D_in G"]:
        d = (x[..., None] - self.grid) / self.sigma
        return jnp.exp(-0.5 * d ** 2)

    def __call__(
        self, x: Float[Array, "B D_in"]
    ) -> Float[Array, "B D_out"]:
        base = nnx.silu(x) @ self.base_w.value
        rbf = self._basis(x)                                                 # (B, in_dim, G)
        spline = jnp.einsum("bif,oif->bo", rbf, self.coeff.value)
        return base + spline


# ---------------------------------------------------------------------------
# Capsule networks
# ---------------------------------------------------------------------------

def squash(
    s: Float[Array, "..."], axis: int = -1, eps: float = 1e-8,
) -> Float[Array, "..."]:
    """Sabour et al. 2017 - non-linear "squash" preserving direction but compressing norm."""
    norm_sq = jnp.sum(s * s, axis=axis, keepdims=True)
    norm = jnp.sqrt(norm_sq) + eps
    return (norm_sq / (1 + norm_sq)) * (s / norm)


class CapsuleLayer(nnx.Module):
    """Fully-connected capsule layer with dynamic routing.

    Input : ``(B, N_in,  D_in)`` capsules
    Output: ``(B, N_out, D_out)`` capsules
    """

    def __init__(self, num_in: int, dim_in: int, num_out: int, dim_out: int,
                 routing_iters: int = 3, *, rngs: nnx.Rngs) -> None:
        self.num_in = num_in
        self.num_out = num_out
        self.iters = routing_iters
        self.W = nnx.Param(
            jax.random.normal(rngs.params(),
                              (num_out, num_in, dim_out, dim_in)) * 0.1)

    def __call__(
        self, x: Float[Array, "B N_in D_in"]
    ) -> Float[Array, "B N_out D_out"]:
        u_hat = jnp.einsum("oidp,bip->boid", self.W.value, x)
        b = jnp.zeros((x.shape[0], self.num_out, self.num_in))
        v = None
        for _ in range(self.iters):
            c = jax.nn.softmax(b, axis=1)
            s = jnp.sum(c[..., None] * u_hat, axis=2)
            v = squash(s, axis=-1)
            b = b + jnp.sum(u_hat * v[:, :, None], axis=-1)
        return v


def dynamic_routing(
    votes: Float[Array, "B O I D"], num_iter: int = 3,
) -> Float[Array, "B O D"]:
    """Standalone dynamic routing on pre-computed votes ``(B, O, I, D)``."""
    B, O, I, _ = votes.shape
    b = jnp.zeros((B, O, I))
    v = None
    for _ in range(num_iter):
        c = jax.nn.softmax(b, axis=1)
        s = jnp.sum(c[..., None] * votes, axis=2)
        v = squash(s, axis=-1)
        b = b + jnp.sum(votes * v[:, :, None], axis=-1)
    return v


# ---------------------------------------------------------------------------
# Slot Attention
# ---------------------------------------------------------------------------

class SlotAttention(nnx.Module):
    """Slot Attention (Locatello et al. 2020) for object-centric learning."""

    def __init__(self, num_slots: int, dim: int, iters: int = 3,
                 hidden_mlp: Optional[int] = None, eps: float = 1e-8,
                 *, rngs: nnx.Rngs) -> None:
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slot_mu = nnx.Param(
            jax.random.normal(rngs.params(), (1, 1, dim)))
        self.slot_log_sigma = nnx.Param(jnp.zeros((1, 1, dim)))

        self.to_q = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.to_k = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        self.to_v = nnx.Linear(dim, dim, use_bias=False, rngs=rngs)
        from .sequence_blocks import GRUCell
        self.gru = GRUCell(dim, dim, rngs=rngs)
        self.fc1 = nnx.Linear(dim, hidden_mlp or 2 * dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_mlp or 2 * dim, dim, rngs=rngs)
        self.norm_input = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_slots = nnx.LayerNorm(dim, rngs=rngs)
        self.norm_pre_ff = nnx.LayerNorm(dim, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B T D"],
        key: PRNGKeyArray,
    ) -> Float[Array, "B K D"]:
        B, _, D = x.shape
        x = self.norm_input(x)
        k, v = self.to_k(x), self.to_v(x)

        slots = (self.slot_mu.value
                 + jnp.exp(self.slot_log_sigma.value)
                 * jax.random.normal(key, (B, self.num_slots, D)))

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale
            attn_logits = jnp.einsum("bsd,bnd->bsn", q, k)
            attn = jax.nn.softmax(attn_logits, axis=1)
            attn = attn / (jnp.sum(attn, axis=-1, keepdims=True) + self.eps)
            updates = jnp.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(updates.reshape(-1, D),
                             slots_prev.reshape(-1, D)).reshape(B,
                                                                self.num_slots, D)
            slots = slots + self.fc2(nnx.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots
