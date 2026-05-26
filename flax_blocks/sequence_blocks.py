"""Section 8 - Sequence-modeling blocks.

Vanilla RNN / LSTM / GRU cells (custom implementations + thin re-exports
of Flax built-ins), a small state-space model and the selective-scan
operator used by Mamba.
"""

from __future__ import annotations

from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float


# ---------------------------------------------------------------------------
# RNN / LSTM / GRU cells
# ---------------------------------------------------------------------------

class RNNCell(nnx.Module):
    """``h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)``."""

    def __init__(self, in_dim: int, hidden: int, *, rngs: nnx.Rngs) -> None:
        self.hidden = hidden
        self.W_ih = nnx.Linear(in_dim, hidden, rngs=rngs)
        self.W_hh = nnx.Linear(hidden, hidden, use_bias=False, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B D_in"],
        h: Float[Array, "B D_hidden"],
    ) -> Float[Array, "B D_hidden"]:
        return jnp.tanh(self.W_ih(x) + self.W_hh(h))


class LSTMCell(nnx.Module):
    """LSTM cell with explicit input/forget/cell/output gates."""

    def __init__(self, in_dim: int, hidden: int, *, rngs: nnx.Rngs) -> None:
        self.hidden = hidden
        self.W = nnx.Linear(in_dim + hidden, 4 * hidden, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B D_in"],
        state: tuple[Float[Array, "B D_hidden"], Float[Array, "B D_hidden"]],
    ) -> tuple[Float[Array, "B D_hidden"], Float[Array, "B D_hidden"]]:
        h, c = state
        gates = self.W(jnp.concatenate([x, h], axis=-1))
        i, f, g, o = jnp.split(gates, 4, axis=-1)
        i, f, o = nnx.sigmoid(i), nnx.sigmoid(f), nnx.sigmoid(o)
        g = jnp.tanh(g)
        c = f * c + i * g
        h = o * jnp.tanh(c)
        return h, c


class GRUCell(nnx.Module):
    """Cho et al. 2014 - simplified gating."""

    def __init__(self, in_dim: int, hidden: int, *, rngs: nnx.Rngs) -> None:
        self.hidden = hidden
        self.x2h = nnx.Linear(in_dim, 3 * hidden, rngs=rngs)
        self.h2h = nnx.Linear(hidden, 3 * hidden, rngs=rngs)

    def __call__(
        self,
        x: Float[Array, "B D_in"],
        h: Float[Array, "B D_hidden"],
    ) -> Float[Array, "B D_hidden"]:
        xz, xr, xn = jnp.split(self.x2h(x), 3, axis=-1)
        hz, hr, hn = jnp.split(self.h2h(h), 3, axis=-1)
        z = nnx.sigmoid(xz + hz)
        r = nnx.sigmoid(xr + hr)
        n = jnp.tanh(xn + r * hn)
        return (1 - z) * n + z * h


def run_recurrent(
    cell: nnx.Module,
    x: Float[Array, "B T D_in"],
    state: Optional[object] = None,
) -> Float[Array, "B T D_hidden"]:
    """Apply ``cell`` along the time dimension of ``x: (B, T, D)``."""
    B, T, _ = x.shape
    if isinstance(cell, LSTMCell):
        if state is None:
            state = (jnp.zeros((B, cell.hidden)), jnp.zeros((B, cell.hidden)))
        outs = []
        for t in range(T):
            state = cell(x[:, t], state)
            outs.append(state[0])
        return jnp.stack(outs, axis=1)
    if state is None:
        state = jnp.zeros((B, cell.hidden))
    outs = []
    for t in range(T):
        state = cell(x[:, t], state)
        outs.append(state)
    return jnp.stack(outs, axis=1)


# ---------------------------------------------------------------------------
# State-space model (S4-style)
# ---------------------------------------------------------------------------

class StateSpaceModel(nnx.Module):
    """Diagonal SSM with learnable continuous parameters, ZOH-discretized."""

    def __init__(self, dim: int, state_dim: int = 64, *, rngs: nnx.Rngs) -> None:
        self.dim = dim
        self.state = state_dim
        log_a = jnp.log(jnp.linspace(0.5, 4.0, state_dim))
        self.A_log = nnx.Param(jnp.broadcast_to(log_a, (dim, state_dim)).copy())
        self.B = nnx.Param(jax.random.normal(rngs.params(), (dim, state_dim)) * 0.02)
        self.C = nnx.Param(jax.random.normal(rngs.params(), (dim, state_dim)) * 0.02)
        self.D = nnx.Param(jnp.zeros((dim,)))
        self.log_dt = nnx.Param(jnp.zeros((dim,)))

    def __call__(
        self, u: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        B, T, D = u.shape
        dt = jnp.exp(self.log_dt.value)
        A = -jnp.exp(self.A_log.value)
        Ad = jnp.exp(A * dt[:, None])                                   # (D, S)
        Bd = (Ad - 1) / A * self.B.value
        Cv = self.C.value

        def step(state, ut):
            new_state = Ad * state + Bd * ut[:, :, None]
            y = jnp.sum(new_state * Cv, axis=-1)
            return new_state, y

        x0 = jnp.zeros((B, D, self.state), dtype=u.dtype)
        _, ys = jax.lax.scan(step, x0, jnp.swapaxes(u, 0, 1))
        return jnp.swapaxes(ys, 0, 1) + self.D.value * u


# ---------------------------------------------------------------------------
# Selective scan (Mamba)
# ---------------------------------------------------------------------------

def selective_scan(
    u: Float[Array, "B T D"],
    delta: Float[Array, "B T D"],
    A: Float[Array, "D N"],
    B: Float[Array, "B T N"],
    C: Float[Array, "B T N"],
) -> Float[Array, "B T D"]:
    """Selective scan: ``x_t = exp(d * A) x_{t-1} + d * B u_t``.

    Shapes: u (B,T,D); delta (B,T,D); A (D,N); B,C (B,T,N).
    """
    bsz, T, D = u.shape
    N = A.shape[1]

    def step(state, t_inputs):
        ut, dt, Bt, Ct = t_inputs
        Ad = jnp.exp(dt[:, :, None] * A[None])                          # (B,D,N)
        Bd = dt[:, :, None] * Bt[:, None, :]
        new_state = Ad * state + Bd * ut[:, :, None]
        y = jnp.sum(new_state * Ct[:, None, :], axis=-1)
        return new_state, y

    x0 = jnp.zeros((bsz, D, N), dtype=u.dtype)
    _, ys = jax.lax.scan(
        step, x0,
        (jnp.swapaxes(u, 0, 1), jnp.swapaxes(delta, 0, 1),
         jnp.swapaxes(B, 0, 1), jnp.swapaxes(C, 0, 1)))
    return jnp.swapaxes(ys, 0, 1)


class MambaBlock(nnx.Module):
    """A small Mamba-style block: depthwise conv + selective scan + gate."""

    def __init__(self, dim: int, state_dim: int = 16, conv_kernel: int = 4,
                 expand: int = 2, *, rngs: nnx.Rngs) -> None:
        inner = dim * expand
        self.inner = inner
        self.state_dim = state_dim
        self.in_proj = nnx.Linear(dim, 2 * inner, rngs=rngs)
        self.conv = nnx.Conv(inner, inner, (conv_kernel,),
                             padding=[(conv_kernel - 1, 0)],
                             feature_group_count=inner, rngs=rngs)
        self.x_proj = nnx.Linear(inner, 2 * state_dim + inner, rngs=rngs)
        self.dt_proj = nnx.Linear(inner, inner, rngs=rngs)
        self.A_log = nnx.Param(
            jnp.broadcast_to(
                jnp.log(jnp.arange(1, state_dim + 1, dtype=jnp.float32)),
                (inner, state_dim)).copy())
        self.D = nnx.Param(jnp.ones((inner,)))
        self.out_proj = nnx.Linear(inner, dim, rngs=rngs)

    def __call__(
        self, x: Float[Array, "B T D"]
    ) -> Float[Array, "B T D"]:
        x_in, gate = jnp.split(self.in_proj(x), 2, axis=-1)
        x_conv = self.conv(x_in)
        x_act = nnx.silu(x_conv)

        params = self.x_proj(x_act)
        dt, Bp, Cp = jnp.split(
            params, [self.inner, self.inner + self.state_dim], axis=-1)
        dt = nnx.softplus(self.dt_proj(dt))
        A = -jnp.exp(self.A_log.value)
        y = selective_scan(x_act, dt, A, Bp, Cp) + self.D.value * x_act
        return self.out_proj(y * nnx.silu(gate))
