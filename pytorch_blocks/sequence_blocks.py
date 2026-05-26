"""Section 8 - Sequence-modeling blocks.

Vanilla RNN / LSTM / GRU cells, a small state-space model and the
selective-scan operator used by Mamba.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from ._typecheck import typecheck
from torch import Tensor


# ---------------------------------------------------------------------------
# RNN / LSTM / GRU cells
# ---------------------------------------------------------------------------

class RNNCell(nn.Module):
    """``h_t = tanh(W_ih x_t + W_hh h_{t-1} + b)``."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.W_ih = nn.Linear(in_dim, hidden)
        self.W_hh = nn.Linear(hidden, hidden, bias=False)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B D_in"],
        h: Float[Tensor, "B D_h"],
    ) -> Float[Tensor, "B D_h"]:
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


class LSTMCell(nn.Module):
    """LSTM cell with explicit input / forget / cell / output gates."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.W = nn.Linear(in_dim + hidden, 4 * hidden)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B D_in"],
        state: tuple[Float[Tensor, "B D_h"], Float[Tensor, "B D_h"]],
    ) -> tuple[Float[Tensor, "B D_h"], Float[Tensor, "B D_h"]]:
        h, c = state
        gates = self.W(torch.cat([x, h], dim=-1))
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class GRUCell(nn.Module):
    """Cho et al. 2014 - simplified gating."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.x2h = nn.Linear(in_dim, 3 * hidden)
        self.h2h = nn.Linear(hidden, 3 * hidden)

    @typecheck
    def forward(
        self,
        x: Float[Tensor, "B D_in"],
        h: Float[Tensor, "B D_h"],
    ) -> Float[Tensor, "B D_h"]:
        xz, xr, xn = self.x2h(x).chunk(3, dim=-1)
        hz, hr, hn = self.h2h(h).chunk(3, dim=-1)
        z = torch.sigmoid(xz + hz)
        r = torch.sigmoid(xr + hr)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


@typecheck
def run_recurrent(
    cell: nn.Module,
    x: Float[Tensor, "B T D"],
    state: Optional[object] = None,
) -> Float[Tensor, "B T D_h"]:
    """Apply ``cell`` along the time dimension of ``x: (B, T, D)``."""
    outs = []
    for t in range(x.shape[1]):
        if isinstance(cell, LSTMCell):
            if state is None:
                state = (torch.zeros(x.shape[0], cell.hidden, device=x.device),
                         torch.zeros(x.shape[0], cell.hidden, device=x.device))
            state = cell(x[:, t], state)
            outs.append(state[0])
        else:
            if state is None:
                state = torch.zeros(x.shape[0], cell.W_hh.out_features
                                    if isinstance(cell, RNNCell) else cell.hidden,
                                    device=x.device)
            state = cell(x[:, t], state)
            outs.append(state)
    return torch.stack(outs, dim=1)


# ---------------------------------------------------------------------------
# State-space model (S4-style, simple parallel form)
# ---------------------------------------------------------------------------

class StateSpaceModel(nn.Module):
    """Diagonal SSM with learnable continuous parameters.

    Discretization uses zero-order-hold and the recurrence is unrolled in
    Python. For long sequences a CUDA scan would be preferred, but this is
    correct and dependency-free.
    """

    def __init__(self, dim: int, state_dim: int = 64) -> None:
        super().__init__()
        self.dim = dim
        self.state = state_dim
        log_a = torch.log(torch.linspace(0.5, 4.0, state_dim))
        self.A_log = nn.Parameter(log_a.unsqueeze(0).expand(dim, state_dim).contiguous())
        self.B = nn.Parameter(torch.randn(dim, state_dim) * 0.02)
        self.C = nn.Parameter(torch.randn(dim, state_dim) * 0.02)
        self.D = nn.Parameter(torch.zeros(dim))
        self.log_dt = nn.Parameter(torch.zeros(dim))

    @typecheck
    def forward(
        self, u: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        B, T, D = u.shape
        dt = self.log_dt.exp()                                  # (D,)
        A = -self.A_log.exp()                                   # (D, S)
        Ad = torch.exp(A * dt[:, None])                         # (D, S) discretization
        Bd = (Ad - 1) / A * self.B                              # (D, S)
        x = torch.zeros(B, D, self.state, device=u.device, dtype=u.dtype)
        outs = []
        for t in range(T):
            x = Ad * x + Bd * u[:, t, :, None]
            y = (x * self.C).sum(-1)
            outs.append(y)
        y = torch.stack(outs, dim=1)
        return y + self.D * u


# ---------------------------------------------------------------------------
# Selective-scan (Mamba)
# ---------------------------------------------------------------------------

@typecheck
def selective_scan(
    u: Float[Tensor, "B T D"],
    delta: Float[Tensor, "B T D"],
    A: Float[Tensor, "D N"],
    B: Float[Tensor, "B T N"],
    C: Float[Tensor, "B T N"],
) -> Float[Tensor, "B T D"]:
    """Sequential selective scan: ``x_t = exp(d*A) x_{t-1} + d*B u_t``.

    All tensors are time/batch-dependent (the "selective" part).

    Shapes:
        u:    (B, T, D)
        delta:(B, T, D)
        A:    (D, N)            shared across batch/time
        B,C:  (B, T, N)
    """
    bsz, T, D = u.shape
    N = A.shape[1]
    x = torch.zeros(bsz, D, N, device=u.device, dtype=u.dtype)
    outs = []
    for t in range(T):
        d = delta[:, t]                                         # (B, D)
        Ad = torch.exp(d[:, :, None] * A[None])                 # (B, D, N)
        Bd = d[:, :, None] * B[:, t, None, :]                   # (B, D, N)
        x = Ad * x + Bd * u[:, t, :, None]
        y = (x * C[:, t, None, :]).sum(-1)                      # (B, D)
        outs.append(y)
    return torch.stack(outs, dim=1)


class MambaBlock(nn.Module):
    """A small Mamba-style block: depthwise conv + selective scan + gate."""

    def __init__(self, dim: int, state_dim: int = 16, conv_kernel: int = 4,
                 expand: int = 2) -> None:
        super().__init__()
        inner = dim * expand
        self.in_proj = nn.Linear(dim, 2 * inner)
        self.conv = nn.Conv1d(inner, inner, conv_kernel, padding=conv_kernel - 1,
                              groups=inner)
        self.x_proj = nn.Linear(inner, 2 * state_dim + inner)
        self.dt_proj = nn.Linear(inner, inner)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32))
            .repeat(inner, 1))
        self.D = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim)
        self.state_dim = state_dim

    @typecheck
    def forward(
        self, x: Float[Tensor, "B T D"]
    ) -> Float[Tensor, "B T D"]:
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_in, gate = xz.chunk(2, dim=-1)
        x_conv = self.conv(x_in.transpose(1, 2))[..., :T].transpose(1, 2)
        x_act = F.silu(x_conv)

        params = self.x_proj(x_act)
        dt, Bp, Cp = params.split([x_act.shape[-1], self.state_dim, self.state_dim], -1)
        dt = F.softplus(self.dt_proj(dt))
        A = -self.A_log.exp()
        y = selective_scan(x_act, dt, A, Bp, Cp) + self.D * x_act
        return self.out_proj(y * F.silu(gate))
