"""Section 17 - Specialized blocks.

Neural ODE (explicit Euler), 2-D Fourier Neural Operator, Kolmogorov-
Arnold Network layer (B-spline based), Capsule layer + dynamic
routing, Slot attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Complex, Float
from torch import Tensor


# ---------------------------------------------------------------------------
# Neural ODE
# ---------------------------------------------------------------------------

class NeuralODE(nn.Module):
    """Continuous-depth network: integrate ``dh/dt = f_theta(h, t)`` with explicit Euler.

    For most research code you'd use :mod:`torchdiffeq`; this is a torch-only
    fallback that's still differentiable end-to-end.
    """

    def __init__(self, dynamics: nn.Module, num_steps: int = 16,
                 t0: float = 0.0, t1: float = 1.0) -> None:
        super().__init__()
        self.dynamics = dynamics
        self.num_steps = num_steps
        self.t0 = t0
        self.t1 = t1

    def forward(
        self, x: Float[Tensor, "B ..."]
    ) -> Float[Tensor, "B ..."]:
        dt = (self.t1 - self.t0) / self.num_steps
        t = torch.full((x.shape[0],), self.t0, device=x.device, dtype=x.dtype)
        h = x
        for _ in range(self.num_steps):
            h = h + dt * self.dynamics(h, t)
            t = t + dt
        return h


# ---------------------------------------------------------------------------
# Fourier Neural Operator (2-D)
# ---------------------------------------------------------------------------

class SpectralConv2d(nn.Module):
    """The core spectral conv used inside FNO.

    Multiplies the lowest-frequency Fourier modes by a learned complex weight
    tensor and zero-keeps the higher modes.
    """

    def __init__(self, in_ch: int, out_ch: int, modes_h: int, modes_w: int) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.mh = modes_h
        self.mw = modes_w
        scale = 1.0 / (in_ch * out_ch)
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_h, modes_w,
                                                   dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_h, modes_w,
                                                   dtype=torch.cfloat))

    @staticmethod
    def _compl_mul(
        a: Complex[Tensor, "B C_in M_h M_w"],
        b: Complex[Tensor, "C_in C_out M_h M_w"],
    ) -> Complex[Tensor, "B C_out M_h M_w"]:
        return torch.einsum("bixy,ioxy->boxy", a, b)

    def forward(
        self, x: Float[Tensor, "B C_in H W"]
    ) -> Float[Tensor, "B C_out H W"]:
        B, _, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(B, self.out_ch, H, W // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[..., : self.mh, : self.mw] = self._compl_mul(
            x_ft[..., : self.mh, : self.mw], self.w1)
        out_ft[..., -self.mh:, : self.mw] = self._compl_mul(
            x_ft[..., -self.mh:, : self.mw], self.w2)
        return torch.fft.irfft2(out_ft, s=(H, W), norm="ortho")


class FNOBlock(nn.Module):
    """Spectral conv + pointwise residual + nonlinearity (Li et al. 2020)."""

    def __init__(self, channels: int, modes_h: int = 12, modes_w: int = 12) -> None:
        super().__init__()
        self.spectral = SpectralConv2d(channels, channels, modes_h, modes_w)
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B C H W"]:
        return F.gelu(self.spectral(x) + self.skip(x))


# ---------------------------------------------------------------------------
# Kolmogorov-Arnold Network layer
# ---------------------------------------------------------------------------

class KANLayer(nn.Module):
    """A single layer of a Kolmogorov-Arnold Network (Liu et al. 2024).

    Each edge ``(i, j)`` carries a learnable function expressed as
    ``b * silu(x) + sum_k coeff_k * B_k(x)`` over fixed B-spline basis points.
    """

    def __init__(self, in_dim: int, out_dim: int, num_grid: int = 8,
                 grid_range: tuple[float, float] = (-1.0, 1.0)) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_grid = num_grid
        grid = torch.linspace(grid_range[0], grid_range[1], num_grid)
        self.register_buffer("grid", grid, persistent=False)
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, num_grid) * 0.1)
        self.base_w = nn.Parameter(torch.randn(out_dim, in_dim) * 0.1)
        sigma = (grid_range[1] - grid_range[0]) / (num_grid - 1)
        self.register_buffer("sigma", torch.tensor(sigma), persistent=False)

    def _basis(
        self, x: Float[Tensor, "B D_in"]
    ) -> Float[Tensor, "B D_in G"]:
        # Gaussian RBF basis - simpler and differentiable substitute for B-splines
        d = (x[..., None] - self.grid) / self.sigma
        return torch.exp(-0.5 * d ** 2)                                     # (..., G)

    def forward(
        self, x: Float[Tensor, "B D_in"]
    ) -> Float[Tensor, "B D_out"]:
        base = F.linear(F.silu(x), self.base_w)
        rbf = self._basis(x)                                                # (B, in_dim, G)
        spline = torch.einsum("bif,oif->bo", rbf, self.coeff)
        return base + spline


# ---------------------------------------------------------------------------
# Capsule networks
# ---------------------------------------------------------------------------

def squash(
    s: Float[Tensor, "..."], dim: int = -1, eps: float = 1e-8
) -> Float[Tensor, "..."]:
    """Sabour et al. 2017 - non-linear "squash" that keeps direction but compresses norm."""
    norm_sq = s.pow(2).sum(dim=dim, keepdim=True)
    norm = norm_sq.sqrt() + eps
    return (norm_sq / (1 + norm_sq)) * (s / norm)


class CapsuleLayer(nn.Module):
    """Fully-connected capsule layer with dynamic routing.

    Input:  ``(B, N_in,  D_in)``  capsules
    Output: ``(B, N_out, D_out)`` capsules
    """

    def __init__(self, num_in: int, dim_in: int, num_out: int, dim_out: int,
                 routing_iters: int = 3) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_out = num_out
        self.iters = routing_iters
        self.W = nn.Parameter(torch.randn(num_out, num_in, dim_out, dim_in) * 0.1)

    def forward(
        self, x: Float[Tensor, "B N_in D_in"]
    ) -> Float[Tensor, "B N_out D_out"]:
        B = x.shape[0]
        u_hat = torch.einsum("oidp,bip->boid", self.W, x)                   # (B, O, I, D_out)
        b = torch.zeros(B, self.num_out, self.num_in, device=x.device)
        for _ in range(self.iters):
            c = b.softmax(dim=1)
            s = (c.unsqueeze(-1) * u_hat).sum(dim=2)                        # (B, O, D_out)
            v = squash(s, dim=-1)
            b = b + (u_hat * v.unsqueeze(2)).sum(dim=-1)
        return v


def dynamic_routing(
    votes: Float[Tensor, "B O I D"], num_iter: int = 3
) -> Float[Tensor, "B O D"]:
    """Standalone dynamic routing on pre-computed votes ``(B, O, I, D)``."""
    B, O, I, _ = votes.shape
    b = torch.zeros(B, O, I, device=votes.device)
    for _ in range(num_iter):
        c = b.softmax(dim=1)
        s = (c.unsqueeze(-1) * votes).sum(dim=2)
        v = squash(s, dim=-1)
        b = b + (votes * v.unsqueeze(2)).sum(dim=-1)
    return v


# ---------------------------------------------------------------------------
# Slot Attention
# ---------------------------------------------------------------------------

class SlotAttention(nn.Module):
    """Slot Attention (Locatello et al. 2020) for object-centric learning."""

    def __init__(self, num_slots: int, dim: int, iters: int = 3,
                 hidden_mlp: int | None = None, eps: float = 1e-8) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slot_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_mlp or 2 * dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_mlp or 2 * dim, dim))
        self.norm_input = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(
        self, x: Float[Tensor, "B N D"]
    ) -> Float[Tensor, "B K D"]:
        B, N, D = x.shape
        x = self.norm_input(x)
        k, v = self.to_k(x), self.to_v(x)

        slots = self.slot_mu + self.slot_log_sigma.exp() * \
            torch.randn(B, self.num_slots, D, device=x.device)

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots)) * self.scale
            attn_logits = torch.einsum("bsd,bnd->bsn", q, k)
            attn = attn_logits.softmax(dim=1)                               # softmax over slots
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.eps)
            updates = torch.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(updates.reshape(-1, D),
                             slots_prev.reshape(-1, D)).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))
        return slots
